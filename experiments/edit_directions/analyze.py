"""Summarize run.py's results: success by layer, bootstrap CIs, paired tests, plots.

    python experiments/edit_directions/analyze.py experiments/edit_directions/results/*.jsonl

For each model, task and edit, at one strength (push: alpha 1; swap: strength 2,
change with --push-strength / --swap-strength), it reports each kind's best
layer and success rate there, each kind's rate at the other kind's best layer,
and an exact McNemar test between ``unembed`` and ``jlens`` at both of those
layers. Pass diagnose.py's output too, and swaps are also compared with
activation patching: the ceiling, and the layer where later tokens have read
the entity. Writes ``summary.md`` and one ``<model>.png`` next to the first input.

    python experiments/edit_directions/analyze.py experiments/edit_directions/results/google__gemma-3-4b-it.jsonl \
        experiments/edit_directions/results/diagnose-google__gemma-3-4b-it.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

KINDS = ("unembed", "jlens", "random")
COLORS = {"unembed": "#d95f02", "jlens": "#1b9e77", "random": "#999999"}


def load(paths):
    """run.py's rows and meta lines, and diagnose.py's rows (no ``edit`` field)."""
    rows, meta, diag = [], [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("meta"):
                meta.append(r)
            elif "edit" in r:
                rows.append(r)
            else:  # older diagnose files don't record the model
                r.setdefault("model", Path(p).stem.removeprefix("diagnose-").replace("__", "/"))
                diag.append(r)
    return rows, meta, diag


def patching(diag):
    """Per (model, task): success by layer of each diagnose.py kind, and the handoff layer.

    The handoff is the first layer where copying the cells after the entity beats copying the entity's own cell:
    by then later tokens have read the entity, so no edit of the entity cell can work.
    """
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in diag:
        by[r["model"], r["task"]][r["kind"]][r["layer"]].append(r["success"])
    out = {}
    for key, kinds in by.items():
        rate = {k: {l: float(np.mean(v)) for l, v in sorted(ls.items())} for k, ls in kinds.items()}
        ent, aft = rate["patch_entity"], rate["patch_after"]
        handoff = next((l for l in ent if aft[l] > ent[l]), None)
        n = len(next(iter(kinds["patch_entity"].values())))
        out[key] = dict(rate=rate, handoff=handoff, n=n)
    return out


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired successes: ``(a only, b only, p)``."""
    keys = a.keys() & b.keys()
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
    return only_a, only_b, p


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", nargs="+")
    p.add_argument("--push-strength", type=float, default=1.0)
    p.add_argument("--swap-strength", type=float, default=2.0)
    a = p.parse_args()
    rows, meta, diag = load(a.results)
    patch = patching(diag)
    strength = {"push": a.push_strength, "swap": a.swap_strength}
    out_dir = Path(a.results[0]).parent

    # by[(model, task, edit)][kind][layer] = {pair key: success}
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        if r["strength"] != strength[r["edit"]]:
            continue
        key = f"{r['item']}|{r.get('target', r.get('new'))}"
        by[r["model"], r["task"], r["edit"]][r["kind"]][r["layer"]][key] = r["success"]

    lines = ["# Edit directions: logit lens (unembed) vs J-lens", "",
             f"Strength: push alpha = {a.push_strength} × |cell|, swap strength = {a.swap_strength}.", ""]
    lines += [f"- {m['model']} {m['task']}: {m['n_one_token']} one-token items, {m.get('n_checked', m['n_one_token'])} checked, {m['n_correct']} answered correctly"
              for m in meta]
    lines += ["", "| model | task | edit | n | unembed best (layer) | jlens best (layer) | random best | "
             "unembed @ jlens's layer | jlens @ unembed's layer | McNemar p @ jlens's / unembed's layer |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    curves = defaultdict(dict)
    for (model, task, edit), kinds in sorted(by.items()):
        rate = {k: {l: np.mean(list(v.values())) for l, v in kinds[k].items()} for k in KINDS if k in kinds}
        best = {k: max(r, key=r.get) for k, r in rate.items()}
        n = len(next(iter(kinds["jlens"].values())))
        u, j = best["unembed"], best["jlens"]
        p_at_j, p_at_u = mcnemar(kinds["unembed"][j], kinds["jlens"][j])[2], mcnemar(kinds["unembed"][u], kinds["jlens"][u])[2]
        pct = lambda x: f"{100 * x:.0f}%"
        lines.append(f"| {model} | {task} | {edit} | {n} | {pct(rate['unembed'][u])} ({u}) | {pct(rate['jlens'][j])} ({j}) | "
                     f"{pct(max(rate['random'].values())) if 'random' in rate else '-'} | {pct(rate['unembed'][j])} | "
                     f"{pct(rate['jlens'][u])} | {p_at_j:.2g} / {p_at_u:.2g} |")
        for k in rate:
            layers = sorted(kinds[k])
            ci = [bootstrap_ci(np.array(list(kinds[k][l].values()), float)) for l in layers]
            curves[model][task, edit, k] = (layers, [rate[k][l] for l in layers], ci)

    if patch:
        lines += ["", "## Swaps vs activation patching (diagnose.py)", "",
                  "Ceiling: the entity cell copied from a prompt naming the new entity. Handoff: the first layer where "
                  "copying the cells after the entity beats copying the entity cell. Swap rates here are on "
                  "diagnose.py's own pairs, at its strength.", "",
                  "| model | task | n | ceiling (layers before handoff) | handoff layer | jlens swap best (layer) | "
                  "ceiling there | jlens / ceiling | unembed swap best (layer) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for (model, task), d in sorted(patch.items()):
            r, h = d["rate"], d["handoff"]
            before = [l for l in r["patch_entity"] if h is None or l < h]
            ceil = np.mean([r["patch_entity"][l] for l in before])
            j = max(r["jlens"], key=r["jlens"].get)
            u = max(r["unembed"], key=r["unembed"].get)
            c = r["patch_entity"][j]
            pct = lambda x: f"{100 * x:.0f}%"
            lines.append(f"| {model} | {task} | {d['n']} | {pct(ceil)} | {h} | {pct(r['jlens'][j])} ({j}) | {pct(c)} | "
                         f"{pct(r['jlens'][j] / c) if c else '-'} | {pct(r['unembed'][u])} ({u}) |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model, c in curves.items():
        tasks = sorted({t for t, _, _ in c})
        edits = sorted({e for _, e, _ in c})
        fig, axes = plt.subplots(len(tasks), len(edits), figsize=(4.2 * len(edits), 2.6 * len(tasks)),
                                 squeeze=False, sharex=True, sharey=True)
        for i, t in enumerate(tasks):
            for jx, e in enumerate(edits):
                ax = axes[i, jx]
                for k in KINDS:
                    if (t, e, k) not in c:
                        continue
                    layers, ys, ci = c[t, e, k]
                    ax.plot(layers, ys, color=COLORS[k], label=k, lw=1.8)
                    ax.fill_between(layers, [lo for lo, _ in ci], [hi for _, hi in ci], color=COLORS[k], alpha=0.15, lw=0)
                d = patch.get((model, t))
                if e == "swap" and d:
                    n = d["n"]
                    for k, ls, label in (("patch_entity", "-", f"patch entity cell (n={n})"),
                                         ("patch_after", "--", f"patch later tokens (n={n})")):
                        r = d["rate"][k]
                        ax.plot(list(r), list(r.values()), color="black", ls=ls, lw=1, label=label)
                    if d["handoff"] is not None:
                        ax.axvline(d["handoff"], color="black", ls=":", lw=1, label="handoff")
                ax.set_title(f"{t}: {e}", fontsize=9)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("layer")
        for ax in axes[:, 0]:
            ax.set_ylabel("success")
        handles, labels = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=len(labels))
        fig.suptitle(model, fontsize=10)
        fig.tight_layout(rect=(0, 0.025, 1, 1))
        path = out_dir / f"{model.replace('/', '__')}.png"
        fig.savefig(path, dpi=150)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
