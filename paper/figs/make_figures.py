"""Paper figures from the experiment results.

    .venv/bin/python paper/figs/make_figures.py

Reads the raw JSONL results under experiments/ (gitignored: rerun the
experiments to regenerate them) and writes PDFs next to this file.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
EXP = HERE.parent.parent / "experiments"
LL, JL, RND, INK, MUTED = "#d95f02", "#1b9e77", "#9a9a9a", "#222222", "#666666"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.quantile(means, 0.025), np.quantile(means, 0.975)


def by_layer(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["layer"]].append(r["success"])
    layers = sorted(d)
    ci = [bootstrap_ci(d[l]) for l in layers]
    return np.array(layers), np.array([np.mean(d[l]) for l in layers]), np.array(ci)


def curve(ax, rows, color, label, lw=1.6, ls="-", band=True):
    layers, y, ci = by_layer(rows)
    if band:
        ax.fill_between(layers, ci[:, 0], ci[:, 1], color=color, alpha=0.18, lw=0)
    ax.plot(layers, y, color=color, lw=lw, ls=ls, label=label, solid_capstyle="round")
    return layers, y


def style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", color=INK, pad=4)
    ax.set_xlim(0, 34)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("layer")
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])


def directions(task="country-capital", model="google__gemma-3-4b-it"):
    """Fig. 2: push and swap success by layer for one task, and the cosine between the two directions."""
    res = EXP / "edit_directions" / "results"
    rows = [json.loads(l) for l in (res / f"{model}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if not r.get("meta") and r["task"] == task]
    diag = [json.loads(l) for l in (res / f"diagnose-{model}.jsonl").read_text().splitlines()]
    diag = [r for r in diag if r["task"] == task]
    cos = json.loads((res / "direction_cosine.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))

    ax = axes[0]
    push = [r for r in rows if r["edit"] == "push" and r["strength"] == 1.0]
    for kind, color, label in (("random", RND, "random"), ("jlens", JL, "J-lens direction"),
                               ("unembed", LL, "logit-lens direction")):
        curve(ax, [r for r in push if r["kind"] == kind], color, label, band=kind != "random")
    style(ax, "(a) Push the answer token", "edits that succeed")
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1]
    swap = [r for r in rows if r["edit"] == "swap" and r["strength"] == 2.0]
    ent = [r for r in diag if r["kind"] == "patch_entity"]
    aft = [r for r in diag if r["kind"] == "patch_after"]
    le, ye = curve(ax, ent, INK, "copy entity cell (ceiling)", lw=0.9, band=False)
    la, ya = curve(ax, aft, INK, "copy cells after it", lw=0.9, ls=(0, (3, 2)), band=False)
    handoff = next(l for l, e, a in zip(le, ye, ya) if a > e)
    ax.axvline(handoff, color=MUTED, lw=0.6, ls=":")
    ax.text(handoff + 0.6, 0.5, f"handoff\n(layer {handoff})", color=MUTED, fontsize=6.5, va="center")
    for kind, color, label in (("random", RND, "random"), ("unembed", LL, "logit-lens direction"),
                               ("jlens", JL, "J-lens direction")):
        curve(ax, [r for r in swap if r["kind"] == kind], color, label, band=kind != "random")
    style(ax, "(b) Swap the country token", None)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[2]
    layers = sorted(int(l) for l in cos["layers"])[:-1]  # the top layer's J is the identity
    m = [cos["layers"][str(l)]["mean"] for l in layers]
    q = [(cos["layers"][str(l)]["q25"], cos["layers"][str(l)]["q75"]) for l in layers]
    ax.fill_between(layers, [a for a, _ in q], [b for _, b in q], color=MUTED, alpha=0.18, lw=0)
    ax.plot(layers, m, color=INK, lw=1.4)
    style(ax, "(c) Agreement of the two directions", None)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("cosine, same word")

    handles = {}
    for a in axes[:2]:
        for h, lab in zip(*a.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = ["logit-lens direction", "J-lens direction", "random", "copy entity cell (ceiling)", "copy cells after it"]
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.6)
    out = HERE / "directions.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out} (handoff at layer {handoff})")


def twohop(template="us", strength=4.0, model="google__gemma-3-4b-it", k=5):
    """Fig. 3: when the state is readable at the city and last tokens, and swapping it at each."""
    rows = [json.loads(l) for l in (EXP / "two_hop" / "results" / f"{model}.jsonl").read_text().splitlines()]
    read = [r for r in rows if r.get("type") == "readout" and r["template"] == template and r["layer"] < 34]
    edits = [r for r in rows if r.get("type") == "edit" and r["template"] == template]

    def readable(where, lens, word):  # rows shaped for curve(): success = in the lens's top k
        return [dict(layer=r["layer"], success=r[word] < k) for r in read if r["where"] == where and r["lens"] == lens]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))

    ax = axes[0]
    curve(ax, readable("city", "jlens", "ctrl_state"), RND, "another item's state", band=False)
    curve(ax, readable("city", "logit", "state"), LL, "state (logit lens)")
    curve(ax, readable("city", "jlens", "state"), JL, "state (J-lens)")
    style(ax, "(a) Read the city token", f"items with word in top {k}")

    ax = axes[1]
    curve(ax, readable("last", "jlens", "ctrl_state"), RND, "another item's state", band=False)
    curve(ax, readable("last", "jlens", "state"), JL, "state (J-lens)")
    curve(ax, readable("last", "jlens", "capital"), JL, "capital (J-lens)", ls=(0, (3, 2)))
    style(ax, "(b) Read the last token", None)

    pc = [r for r in edits if r["edit"] == "patch_city"]
    pl = [r for r in edits if r["edit"] == "patch_last"]
    lc, yc, _ = by_layer(pc)
    _, yl, _ = by_layer(pl)
    handoff = next(l for l, c, x in zip(lc, yc, yl) if x > c)
    for ax, edit, patch, title in ((axes[2], "state@city", pc, "(c) Swap state at city"),
                                   (axes[3], "state@last", pl, "(d) Swap state at last token")):
        curve(ax, patch, INK, "copy that cell (ceiling)", lw=0.9, band=False)
        ax.axvline(handoff, color=MUTED, lw=0.6, ls=":")
        for kind, color, label in (("unembed", LL, "logit-lens direction"), ("jlens", JL, "J-lens direction")):
            curve(ax, [r for r in edits if r["edit"] == edit and r.get("kind") == kind and r["strength"] == strength],
                  color, label)
        style(ax, title, "edits that succeed" if edit == "state@city" else None)
    axes[3].text(handoff - 1.4, 0.5, f"handoff\n(layer {handoff})", color=MUTED, fontsize=6.5, va="center", ha="right")
    curve(axes[3], [r for r in edits if r["edit"] == "random@last" and r["strength"] == strength], RND, "random",
          band=False)
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)

    orders = (["state (J-lens)", "state (logit lens)", "capital (J-lens)", "another item's state"],
              ["J-lens direction", "logit-lens direction", "random", "copy that cell (ceiling)"])
    for group, x, order in ((axes[:2], 0.26, orders[0]), (axes[2:], 0.76, orders[1])):
        handles = {}
        for a in group:
            for h, lab in zip(*a.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        fig.legend([handles[o] for o in order], order, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(x, -0.03), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    out = HERE / "twohop.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out} (handoff at layer {handoff})")


if __name__ == "__main__":
    directions()
    twohop()
