"""Summarize two_hop/run.py: when the state and capital become readable, and which edits move the answer.

    python experiments/two_hop/analyze.py experiments/two_hop/results/google__gemma-3-4b-it.jsonl

Readout: per layer, the fraction of items whose state / capital / control words
are in a lens's top ``--k`` at the city token and the last token, and per item,
the first layer each gets there. Edits: success by layer for each edit and
direction. Writes ``summary.md``, ``<model>-readout.png`` and ``<model>-edits.png``
next to the input.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WORDS = {"state": "#1f78b4", "capital": "#e66101", "ctrl_state": "#a6cee3", "ctrl_capital": "#fdb863"}
LENS_STYLE = {"logit": "--", "jlens": "-"}
EDIT_COLORS = {"unembed": "#d95f02", "jlens": "#1b9e77", "random": "#999999"}


def first(layers_ranks: list[tuple[int, int]], k: int) -> int | None:
    return next((l for l, r in sorted(layers_ranks) if r < k), None)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results")
    p.add_argument("--k", type=int, default=5, help="'readable' = in the lens's top k")
    p.add_argument("--strength", type=float, default=4.0, help="swap strength to plot")
    a = p.parse_args()
    path = Path(a.results)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    meta = [r for r in rows if r.get("meta")]
    read = [r for r in rows if r.get("type") == "readout"]
    edits = [r for r in rows if r.get("type") == "edit"]
    model = meta[0]["model"]
    templates = [m["template"] for m in meta]
    lines = [f"# Two-hop recall: {model}", "", f"Readable = in the lens's top {a.k}.", ""]
    for m in meta:
        lines.append(f"- {m['template']}: {m['n_items']} one-token items, {m['n_correct']} answered correctly. "
                     f"Wrong answers: {', '.join(f'{c} → {top.strip()}' for c, _, top in m['wrong'])}")
    lines.append("")

    # --- readout
    by = defaultdict(lambda: defaultdict(list))  # (template, where, lens) -> word -> [(layer, rank)] per item
    per_item = defaultdict(lambda: defaultdict(list))  # (template, where, lens, city) -> word -> [(layer, rank)]
    for r in read:
        for w in WORDS:
            by[r["template"], r["where"], r["lens"]][w, r["layer"]].append(r[w] < a.k)
            per_item[r["template"], r["where"], r["lens"], r["city"]][w].append((r["layer"], r[w]))
    lines += ["## When each word becomes readable (per item, first layer in the top k)", "",
              "| template | token | lens | n | state ever | capital ever | ctrl state ever | state before capital | "
              "state same layer | capital before state | median first layer: state / capital |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in templates:
        for where in ("city", "last"):
            for lens in ("logit", "jlens"):
                items = {k[3]: v for k, v in per_item.items() if k[:3] == (t, where, lens)}
                if not items:
                    continue
                f_s = {c: first(v["state"], a.k) for c, v in items.items()}
                f_c = {c: first(v["capital"], a.k) for c, v in items.items()}
                f_x = {c: first(v["ctrl_state"], a.k) for c, v in items.items()}
                n = len(items)
                both = [c for c in items if f_s[c] is not None and f_c[c] is not None]
                pct = lambda x: f"{100 * x / n:.0f}%"
                med = lambda d: np.median([v for v in d.values() if v is not None]) if any(v is not None for v in d.values()) else float("nan")
                lines.append(f"| {t} | {where} | {lens} | {n} | {pct(sum(v is not None for v in f_s.values()))} | "
                             f"{pct(sum(v is not None for v in f_c.values()))} | {pct(sum(v is not None for v in f_x.values()))} | "
                             f"{pct(sum(f_s[c] < f_c[c] for c in both))} | {pct(sum(f_s[c] == f_c[c] for c in both))} | "
                             f"{pct(sum(f_s[c] > f_c[c] for c in both))} | {med(f_s):.0f} / {med(f_c):.0f} |")
    lines.append("")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(templates), 2, figsize=(9, 2.8 * len(templates)), squeeze=False, sharex=True, sharey=True)
    for i, t in enumerate(templates):
        for j, where in enumerate(("city", "last")):
            ax = axes[i, j]
            for lens, ls in LENS_STYLE.items():
                d = by[t, where, lens]
                for w, color in WORDS.items():
                    ls_ = sorted(l for ww, l in d if ww == w)
                    ax.plot(ls_, [np.mean(d[w, l]) for l in ls_], color=color, ls=ls, lw=1.6,
                            label=f"{w.replace('_', ' ')} ({'logit lens' if lens == 'logit' else 'J-lens'})")
            ax.set_title(f"{t}: {'city token' if where == 'city' else 'last token'}", fontsize=9)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("layer")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"fraction in top {a.k}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=4)
    fig.suptitle(f"{model}: when the state and capital are readable", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path.with_name(f"{model.replace('/', '__')}-readout.png"), dpi=150)

    # --- edits
    if edits:
        succ = defaultdict(lambda: defaultdict(list))  # (template, edit, kind, strength) -> layer -> [success]
        for r in edits:
            succ[r["template"], r["edit"], r.get("kind", "patch"), r.get("strength")][r["layer"]].append(r["success"])
        lines += ["## Edits (success = the new capital is top-1)", "",
                  "| template | edit | direction | strength | n | best (layer) |", "| --- | --- | --- | --- | --- | --- |"]
        for key in sorted(succ, key=lambda k: (k[0], k[1], k[2], k[3] or 0)):
            rate = {l: np.mean(v) for l, v in succ[key].items()}
            b = max(rate, key=rate.get)
            n = len(next(iter(succ[key].values())))
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3] or '-'} | {n} | {100 * rate[b]:.0f}% ({b}) |")
        cf = defaultdict(list)
        for r in edits:
            cf[r["template"]].append(r["cf_success"])
        lines += ["", "Counterfactual prompt answered with the new capital: " +
                  ", ".join(f"{t} {100 * np.mean(v):.0f}%" for t, v in cf.items()), ""]

        panels = [("state@last", "swap state at last token"), ("state@city", "swap state at city token"),
                  ("city@city", "swap city at city token")]
        fig, axes = plt.subplots(len(templates), 3, figsize=(12, 2.8 * len(templates)), squeeze=False, sharex=True, sharey=True)
        for i, t in enumerate(templates):
            for j, (edit, title) in enumerate(panels):
                ax = axes[i, j]
                patch = "patch_last" if edit == "state@last" else "patch_city"
                d = succ[t, patch, "patch", None]
                ls_ = sorted(d)
                ax.plot(ls_, [np.mean(d[l]) for l in ls_], color="black", lw=1, label=f"patch that cell (ceiling)")
                for kind in ("unembed", "jlens", "random"):
                    e = "random@last" if kind == "random" else edit
                    if kind == "random" and edit != "state@last":
                        continue
                    d = succ[t, e, kind, a.strength]
                    ls_ = sorted(d)
                    ax.plot(ls_, [np.mean(d[l]) for l in ls_], color=EDIT_COLORS[kind], lw=1.8, label=kind)
                ax.set_title(f"{t}: {title}", fontsize=9)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("layer")
        for ax in axes[:, 0]:
            ax.set_ylabel("new capital top-1")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=4)
        fig.suptitle(f"{model}: edits (swap strength {a.strength:g})", fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(path.with_name(f"{model.replace('/', '__')}-edits.png"), dpi=150)

    (path.parent / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
