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
        ax.fill_between(layers, ci[:, 0], ci[:, 1], color=color, alpha=0.32, lw=0)
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
    le, ye = curve(ax, ent, INK, "copy entity cell", lw=0.9, band=False)
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
    ax.fill_between(layers, [a for a, _ in q], [b for _, b in q], color=MUTED, alpha=0.3, lw=0)
    ax.plot(layers, m, color=INK, lw=1.4)
    style(ax, "(c) Agreement of the two directions", None)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("cosine, same word")

    handles = {}
    for a in axes[:2]:
        for h, lab in zip(*a.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = ["logit-lens direction", "J-lens direction", "random", "copy entity cell", "copy cells after it"]
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.6)
    out = HERE / "directions.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out} (handoff at layer {handoff})")


def twohop(template="us", strength=4.0, model="google__gemma-3-4b-it", k=5, column=False, ll_last=True, out=None):
    """Fig. 3: when the state is readable at the city and last tokens, and swapping it at each.

    ``column=True`` lays the four panels out 2 x 2 for one column (twohop_column.pdf). ``ll_last=False`` leaves the
    logit lens out of panel (b).
    """
    rows = [json.loads(l) for l in (EXP / "two_hop" / "results" / f"{model}.jsonl").read_text().splitlines()]
    read = [r for r in rows if r.get("type") == "readout" and r["template"] == template and r["layer"] < 34]
    edits = [r for r in rows if r.get("type") == "edit" and r["template"] == template]

    def readable(where, lens, word):  # rows shaped for curve(): success = in the lens's top k
        return [dict(layer=r["layer"], success=r[word] < k) for r in read if r["where"] == where and r["lens"] == lens]

    if column:  # 2 x 2, each pair of panels with its legend underneath
        fig = plt.figure(figsize=(3.4, 4.1))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 0.3, 1, 0.3], hspace=0.55, wspace=0.08,
                              left=0.16, right=0.98, top=0.95, bottom=0.01)
        axes = [fig.add_subplot(gs[r, c]) for r in (0, 2) for c in (0, 1)]
        legend_axes = [fig.add_subplot(gs[r, :]) for r in (1, 3)]
    else:
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
    if ll_last:
        curve(ax, readable("last", "logit", "state"), LL, "state (logit lens)")
        curve(ax, readable("last", "logit", "capital"), LL, "capital (logit lens)", ls=(0, (3, 2)))
    style(ax, "(b) Read the last token", None)

    pc = [r for r in edits if r["edit"] == "patch_city"]
    pl = [r for r in edits if r["edit"] == "patch_last"]
    lc, yc, _ = by_layer(pc)
    _, yl, _ = by_layer(pl)
    handoff = next(l for l, c, x in zip(lc, yc, yl) if x > c)
    titles = ("(c) Swap at city", "(d) Swap at last token") if column else \
        ("(c) Swap state at city", "(d) Swap state at last token")
    for ax, edit, patch, title in ((axes[2], "state@city", pc, titles[0]), (axes[3], "state@last", pl, titles[1])):
        curve(ax, patch, INK, "copy cell", lw=0.9, band=False)
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
    if column:  # both panels in a row share the y-axis: label it once, on the left
        axes[0].set_ylabel(f"items, word in top {k}")
        axes[2].set_ylabel("edits that succeed")
        for ax in (axes[1], axes[3]):
            ax.tick_params(labelleft=False)

    orders = ((["state (J-lens)", "state (logit lens)", "capital (J-lens)", "capital (logit lens)",
                "another item's state"] if ll_last else
               ["state (J-lens)", "state (logit lens)", "capital (J-lens)", "another item's state"]),
              ["J-lens direction", "logit-lens direction", "random", "copy cell"])
    for g, (group, x, order) in enumerate(((axes[:2], 0.26, orders[0]), (axes[2:], 0.76, orders[1]))):
        handles = {}
        for a in group:
            for h, lab in zip(*a.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        if column:
            legend_axes[g].set_axis_off()
            legend_axes[g].legend([handles[o] for o in order], order, loc="center", ncol=2, frameon=False,
                                  handlelength=2.0, columnspacing=1.2, fontsize=6.5)
        else:
            fig.legend([handles[o] for o in order], order, loc="lower center", ncol=2, frameon=False,
                       bbox_to_anchor=(x, -0.03), handlelength=2.2, columnspacing=1.4)
    if not column:
        fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    out = Path(out) if out else HERE / ("twohop_column.pdf" if column else "twohop.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out} (handoff at layer {handoff})")


def twohop_mechanism(template="us", strength=4.0, model="google__gemma-3-4b-it", k=5):
    """One column: a summary of Scenario B. Two token columns over the layers, with one lane per lens showing where
    the state and the capital are readable (median first layer in the top ``k``), the handoff, and the swap results.
    """
    rows = [json.loads(l) for l in (EXP / "two_hop" / "results" / f"{model}.jsonl").read_text().splitlines()]
    read = sorted((r for r in rows if r.get("type") == "readout" and r["template"] == template and r["layer"] < 34),
                  key=lambda r: r["layer"])
    edits = [r for r in rows if r.get("type") == "edit" and r["template"] == template]
    first = {}
    for r in read:
        for w in ("state", "capital"):
            if r[w] < k:
                first.setdefault((r["city"], r["where"], r["lens"], w), r["layer"])

    def median_first(where, lens, word):
        v = [l for (_, wh, le, wo), l in first.items() if (wh, le, wo) == (where, lens, word)]
        return int(np.median(v)) if len(v) * 2 > len({c for c, *_ in first}) else None  # most items, or none

    def best(edit, kind=None):
        rs = [r for r in edits if r["edit"] == edit and (kind is None or r.get("kind") == kind)
              and r.get("strength", strength) == strength]
        layers, y, _ = by_layer(rs)
        return y.max()

    def best_layer(edit, kind):
        layers, y, _ = by_layer([r for r in edits if r["edit"] == edit and r.get("kind") == kind
                                 and r.get("strength", strength) == strength])
        return int(layers[y.argmax()])

    lc, yc, _ = by_layer([r for r in edits if r["edit"] == "patch_city"])
    _, yl, _ = by_layer([r for r in edits if r["edit"] == "patch_last"])
    handoff = next(l for l, c, x in zip(lc, yc, yl) if x > c)

    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    top, lane = 34, 0.42
    bars = {"city": 1.0, "last": 5.7}  # left edge of each token's pair of lanes
    for where, x0 in bars.items():
        for j, (lens, color) in enumerate((("jlens", JL), ("logit", LL))):
            x = x0 + j * (lane + 0.06)
            ax.add_patch(plt.Rectangle((x, 0), lane, top, facecolor="#f0f0f0", lw=0))
            s, c = median_first(where, lens, "state"), median_first(where, lens, "capital")
            end = c if c is not None else top
            if s is not None:
                ax.add_patch(plt.Rectangle((x, s), lane, end - s, facecolor=color, alpha=0.35, lw=0))
                ax.text(x + lane / 2, (s + end) / 2, "Texas", rotation=90, ha="center", va="center", fontsize=5.5,
                        color=INK)
            if c is not None:
                ax.add_patch(plt.Rectangle((x, c), lane, top - c, facecolor=color, lw=0))
                ax.text(x + lane / 2, (c + top) / 2, "Austin", rotation=90, ha="center", va="center", fontsize=5.5,
                        color="white")
        label = ax.text(x0 - 0.1, top + 0.9, "(a) City token " if where == "city" else "(b) Last token ",
                        ha="left", va="baseline", fontsize=7.5, color=INK)
        ax.annotate('"Dallas"' if where == "city" else '"is"', xy=(1, 0), xycoords=label, ha="left", va="bottom",
                    fontsize=7.5, family="monospace", color=INK)
    for l in (0, 10, 20, 30):
        ax.text(0.75, l, str(l), ha="right", va="center", fontsize=5.5, color=MUTED)
    ax.text(0.15, top / 2, "layer", rotation=90, ha="center", va="center", fontsize=6.5, color=MUTED)

    # The handoff, and the arrow from the city token to the last token.
    ax.plot([0.9, 9.9], [handoff, handoff], color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(9.9, handoff - 0.5, f"handoff (layer {handoff})", ha="right", va="top", fontsize=5.5, color=MUTED)
    ax.annotate("", xy=(bars["last"] - 0.1, handoff + 0.6), xytext=(bars["city"] + 2 * lane + 0.2, handoff - 1.2),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, connectionstyle="arc3,rad=-0.25",
                                mutation_scale=6))
    ax.text((bars["city"] + bars["last"]) / 2 + 0.5, handoff + 2.6, "by the handoff, the last token\nhas copied Texas from Dallas",
            ha="center", va="bottom", fontsize=5.5, color=INK)

    # What each edit does, beside the token it edits, pointing at the layer it was made at.
    bad = "#b03a2e"
    cx, lx = bars["city"] + 2 * lane + 0.45, bars["last"] + 2 * lane + 0.45
    for x, y in ((cx, best_layer("state@city", "jlens")), (lx, 30)):
        ax.annotate("", xy=(x - 0.4, y), xytext=(x - 0.05, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=5))
    y = best_layer("state@city", "jlens")
    ax.text(cx, y, "Swap Texas$\\to$California here:", ha="left", va="center", fontsize=5.8, color=INK)
    ax.text(cx, y - 2.0, "the answer becomes Sacramento", ha="left", va="center", fontsize=5.8, color=INK)
    ax.text(cx, y - 4.2, f"J-lens direction: {best('state@city', 'jlens'):.0%} of pairs", ha="left", va="center",
            fontsize=5.8, color=JL)
    ax.text(cx, y - 6.2, f"logit-lens direction: {best('state@city', 'unembed'):.0%}", ha="left", va="center",
            fontsize=5.8, color=MUTED)
    ax.text(lx, 30, "Swap Texas$\\to$California here:", ha="left", va="center", fontsize=5.8, color=INK)
    ax.text(lx, 28.0, "the answer stays Austin", ha="left", va="center", fontsize=5.8, color=bad)
    ax.text(lx, 26.2, f"(changes in at most {best('state@last', 'jlens'):.0%})", ha="left", va="center",
            fontsize=5.5, color=MUTED)
    for x, side in ((cx, "below"), (lx, "above")):
        ax.text(x, 5.0, f"copying the whole cell\nworks {side} the handoff only", ha="left", va="center",
                fontsize=5.5, color=MUTED)

    # Key: one lane per lens, light for the state, dark for the capital.
    ky = -3.2
    for j, (color, label) in enumerate(((JL, "J-lens"), (LL, "logit lens"))):
        x = 1.0 + j * 3.0
        ax.add_patch(plt.Rectangle((x, ky), 0.35, 1.4, facecolor=color, alpha=0.35, lw=0))
        ax.add_patch(plt.Rectangle((x + 0.4, ky), 0.35, 1.4, facecolor=color, lw=0))
        ax.text(x + 0.9, ky + 0.7, label, ha="left", va="center", fontsize=6, color=INK)
    ax.text(7.0, ky + 0.7, "light: state, dark: capital", ha="left", va="center", fontsize=5.5, color=MUTED)
    ax.set_xlim(0, 10)
    ax.set_ylim(ky - 0.3, top + 3.2)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = HERE / "twohop_mechanism.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Wrote {out} (handoff at layer {handoff})")


def cot_schematic():
    """One column, illustrative (not data): what the lens grid would show at the end of the Sydney question and in
    the reasoning after it, (a) if the model works out the answer in its layers while reading the question, and (b)
    if it only reaches the answer through the steps it writes.
    """
    tokens = ["that", "country", "?", "It", "is", "in", "Australia", ",", "so"]
    n_q, rows = 3, 6  # question tokens shown; layer bands, bottom = early
    # (column, band) -> word read by the lens; bands count from the bottom
    cases = {
        "(a) Answer worked out while reading the question": {
            **{(1, b): "Australia" for b in (2, 3)}, **{(1, b): "Canberra" for b in (4, 5)},
            **{(2, b): "Canberra" for b in (3, 4, 5)}, **{(c, b): "Canberra" for c in (3, 4, 5, 6, 7, 8) for b in (4, 5)}},
        "(b) Answer reached through the written steps": {
            **{(c, b): "Canberra" for c in (6, 7, 8) for b in (4, 5)}},
    }
    word = {"Canberra": (JL, "white"), "Australia": ((*matplotlib.colors.to_rgb(JL), 0.35), INK)}
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.0))
    gap, qgap = 0.1, 0.35  # gap between cells; extra gap between the question and the reasoning
    xs = [c + (qgap if c >= n_q else 0) for c in range(len(tokens))]
    for ax, (title, read) in zip(axes, cases.items()):
        for c, x in enumerate(xs):
            for b in range(rows):
                w = read.get((c, b))
                face, ink = word[w] if w else ("#f0f0f0", INK)
                ax.add_patch(plt.Rectangle((x + gap / 2, b + gap / 2), 1 - gap, 1 - gap, facecolor=face, lw=0))
                if w:
                    ax.text(x + 0.5, b + 0.5, w, ha="center", va="center", fontsize=4.6, color=ink)
            ax.text(x + 0.5, -0.25, tokens[c], ha="center", va="top", fontsize=5.0, family="monospace", color=INK)
        # The model's written answer, after the last reasoning token.
        xa = xs[-1] + 1.25
        ax.annotate("", xy=(xa, rows - 0.5), xytext=(xs[-1] + 1.0, rows - 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=5))
        ax.text(xa + 0.05, rows - 0.5, "writes\nCanberra", ha="left", va="center", fontsize=5.5, color=INK)
        ax.text(-0.15, rows - 0.5, "late", ha="right", va="center", fontsize=5.3, color=MUTED)
        ax.text(-0.15, 0.5, "early", ha="right", va="center", fontsize=5.3, color=MUTED)
        ax.text(-0.15, rows / 2, "layer", ha="right", va="center", fontsize=5.3, color=MUTED)
        ax.text(1.5, rows + 0.2, "end of question", ha="center", va="bottom", fontsize=5.5, color=MUTED)
        ax.text((xs[n_q] + xs[-1] + 1) / 2, rows + 0.2, "reasoning the model writes", ha="center", va="bottom",
                fontsize=5.5, color=MUTED)
        ax.plot([n_q + qgap / 2] * 2, [-0.1, rows + 0.1], color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
        if not any(c < n_q for c, _ in read):
            ax.text(1.5, rows / 2, "Canberra not\nreadable here", ha="center", va="center", fontsize=5.5, color=MUTED)
            ax.text((xs[6] + xs[-1] + 1) / 2, 2.0, "readable once the model\nhas written Australia", ha="center",
                    va="center", fontsize=5.5, color=MUTED)
        ax.set_title(title, loc="left", color=INK, pad=11, fontsize=7.5)
        ax.set_xlim(-1.3, xa + 1.3)
        ax.set_ylim(-1.0, rows + 0.1)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, hspace=0.45)
    out = HERE / "cot_schematic.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Wrote {out}")


def edits_diagram():
    """One column, 2 x 2: the viewer's four edits to a cell h. (a) push and (b) swap act on h along word directions;
    (c) copy and (d) add a difference take cells from other prompts. Illustrative geometry, not data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0))
    arrow = lambda ax, a, b, color, ls="-", lw=1.2: ax.annotate(
        "", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=7,
                                            shrinkA=0, shrinkB=0))
    new_c = "#7570b3"

    ax = axes[0, 0]  # push: h + alpha |h| d_w
    h, d = np.array([1.4, 0.35]), np.array([0.35, 0.94])
    hp = h + 0.8 * np.linalg.norm(h) * d
    arrow(ax, (0, 0), h, INK)
    arrow(ax, (0, 0), d * 0.6, JL, lw=1.0)
    arrow(ax, h, hp, JL, ls=(0, (2, 1.5)), lw=0.9)
    arrow(ax, (0, 0), hp, new_c)
    ax.text(*(h + [0.05, -0.18]), "$h$", fontsize=7, color=INK)
    ax.text(*(d * 0.6 + [-0.32, 0.0]), "$\\hat d_w$", fontsize=7, color=JL)
    ax.text(*(h + hp) / 2 + [0.07, -0.05], "$\\alpha\\Vert h\\Vert\\,\\hat d_w$", fontsize=6.5, color=JL)
    ax.text(*(hp + [-0.55, 0.06]), "$h'$", fontsize=7, color=new_c)
    ax.set_title("(a) Push toward word $w$", loc="left", fontsize=7.5, color=INK)

    ax = axes[0, 1]  # swap: remove the old word's component, add s times its size along the new word
    do, dn = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    h = np.array([1.3, 0.45])
    c = h @ do
    rest = h - c * do
    hp = rest + 1.5 * c * dn
    for v, lab, col in ((do, "$\\hat d_o$", MUTED), (dn, "$\\hat d_n$", MUTED)):
        ax.plot([0, 2.0 * v[0]], [0, 2.4 * v[1]], color="#d0d0d0", lw=0.6, zorder=0)
    ax.text(1.95, -0.2, "old word $\\hat d_o$", fontsize=6.2, color=MUTED, ha="right")
    ax.text(0.08, 2.3, "new word $\\hat d_n$", fontsize=6.2, color=MUTED)
    arrow(ax, (0, 0), h, INK)
    ax.plot([h[0], rest[0]], [h[1], rest[1]], color=LL, lw=0.9, ls=(0, (2, 1.5)))
    ax.text(0.45, h[1] + 0.07, "remove $(h\\cdot\\hat d_o)\\,\\hat d_o$", fontsize=6.0, color=LL)
    arrow(ax, rest, hp, JL, ls=(0, (2, 1.5)), lw=0.9)
    ax.text(rest[0] + 0.08, (rest[1] + hp[1]) / 2, "add $s\\,|h\\cdot\\hat d_o|\\,\\hat d_n$",
            fontsize=6.0, color=JL)
    arrow(ax, (0, 0), hp, new_c)
    ax.text(*(h + [0.04, -0.2]), "$h$", fontsize=7, color=INK)
    ax.text(*(hp + [-0.28, -0.12]), "$h'$", fontsize=7, color=new_c)
    ax.set_title("(b) Swap old word for new", loc="left", fontsize=7.5, color=INK)
    for ax in axes[0]:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(-0.4, 2.3)
        ax.set_ylim(-0.3, 2.55)
        ax.set_axis_off()

    def grid(ax, x0, y0, label, mark=None, color=JL, cols=4, rows=3):
        for cx in range(cols):
            for ry in range(rows):
                on = mark == (cx, ry)
                ax.add_patch(plt.Rectangle((x0 + cx * 0.32, y0 + ry * 0.32), 0.28, 0.28, lw=0,
                                           facecolor=color if on else "#ececec"))
        ax.text(x0 + cols * 0.16 - 0.02, y0 - 0.12, label, ha="center", va="top", fontsize=5.8, color=INK)

    ax = axes[1, 0]  # copy one cell from another prompt
    grid(ax, 0.0, 0.3, "other prompt\n(…Japan)", mark=(1, 1), color=new_c)
    grid(ax, 1.55, 0.3, "this prompt\n(…Russia)", mark=(1, 1), color=new_c)
    arrow(ax, (0.0 + 1 * 0.32 + 0.3, 0.3 + 1 * 0.32 + 0.14), (1.55 + 1 * 0.32 - 0.02, 0.3 + 1 * 0.32 + 0.14), INK,
          lw=0.8)
    ax.text(1.4, 1.42, "$h' = h_{\\mathrm{other}}$", ha="center", fontsize=7, color=INK)
    ax.set_title("(c) Copy a cell from another prompt", loc="left", fontsize=7.5, color=INK)

    ax = axes[1, 1]  # add the difference of two prompts' cells
    grid(ax, 0.0, 0.3, "prompt $a$", mark=(2, 1), color=JL, cols=3)
    ax.text(1.08, 0.75, "$-$", ha="center", va="center", fontsize=9, color=INK)
    grid(ax, 1.25, 0.3, "prompt $b$", mark=(2, 1), color=LL, cols=3)
    ax.text(1.08, 1.42, "$h' = h + \\alpha\\,(h_a - h_b)$", ha="center", fontsize=7, color=INK)
    ax.set_title("(d) Add a difference", loc="left", fontsize=7.5, color=INK)
    for ax in axes[1]:
        ax.set_xlim(-0.1, 2.9)
        ax.set_ylim(-0.35, 1.7)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, hspace=0.35, wspace=0.08)
    out = HERE / "edits.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Wrote {out}")


def teaser(H=2.75):
    """Fig. 1 (full width): the viewer on the Dallas question, from screenshots taken by
    viewer_demo/capture_teaser.mjs. Left: the grid with Texas highlighted (darker = the J-lens ranks Texas higher).
    Middle: the grid after swapping Texas for California at Dallas, layer 21, shaded by how much each cell changed.
    Right: the selected cell's J-lens reading after the edit, and the edit form.
    """
    from PIL import Image
    src = HERE / "viewer_demo" / "teaser"
    S = 3  # device pixels per CSS pixel in the captures
    cells = json.loads((src / "cells.json").read_text())
    cell = {(c["l"], c["p"]): c for c in cells}
    grid_a = np.asarray(Image.open(src / "a_reading_grid.png").convert("RGB"))
    grid_b = np.asarray(Image.open(src / "b_edit_grid.png").convert("RGB"))
    rail = np.asarray(Image.open(src / "b_edit_rail.png").convert("RGB"))
    bar = np.asarray(Image.open(src / "b_edit_bar.png").convert("RGB"))
    grid_a, grid_b = (g[: 1226 * S, : 852 * S] for g in (grid_a, grid_b))
    reads = rail[: 945]  # the selected cell's J-lens reading after the edit
    form = rail[1650:]  # the edit form, without the explanatory lines
    rail = np.concatenate([form[: 130], form[285: 1170], form[1315:]])
    bar = bar[: 205, : 2600]

    gap = 0.12
    wa = grid_a.shape[1] / grid_a.shape[0] * H
    wr = 0.55 * wa  # the form and reading, a little larger than the grids
    hr = rail.shape[0] / rail.shape[1] * wr
    hd = reads.shape[0] / reads.shape[1] * wr
    hb = bar.shape[0] / bar.shape[1] * (wa + gap + wr)
    width = 2 * wa + wr + 2 * gap
    height = H + hb + 0.1
    fig = plt.figure(figsize=(width, height))

    def place(img, x, y, w, h):  # inches from the bottom left
        ax = fig.add_axes([x / width, y / height, w / width, h / height])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        return ax

    ax_a = place(grid_a, 0, 0, wa, H)
    ax_b = place(grid_b, wa + gap, 0, wa, H)
    ax_r = place(rail, 2 * wa + 2 * gap, H - hr, wr, hr)
    ax_d = place(reads, 2 * wa + 2 * gap, H - hr - 0.08 - hd, wr, hd)
    place(bar, wa + gap, H + 0.05, wa + gap + wr, hb)
    hl = "#c0392b"

    def column(ax, p, l0, l1, label, text_side="left"):
        a, b = cell[(l1, p)], cell[(l0, p)]
        x0, x1, y0, y1 = (a["x0"] - 3) * S, (a["x1"] + 3) * S, (a["y0"] - 3) * S, (b["y1"] + 3) * S
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=hl, lw=1.0))
        ax.text(x0 + (x1 - x0) / 2, y0 - 6 * S, label, ha="center", va="bottom", fontsize=7, color=hl,
                fontweight="bold")

    column(ax_a, 9, 6, 34, "(a)")
    column(ax_a, 10, 23, 34, "(b)")
    ax_b.text(cell[(34, 9)]["x0"] * S, -6 * S, "(d)", ha="left", va="bottom", fontsize=7, color=hl, fontweight="bold")
    ax_r.text(rail.shape[1] - 30, 70, "(c)", ha="right", va="center", fontsize=7, color=hl, fontweight="bold")
    ax_d.text(reads.shape[1] - 30, 80, "(c)", ha="right", va="center", fontsize=7, color=hl, fontweight="bold")
    for ax in (ax_a, ax_b):
        ax.set_ylim(ax.get_ylim()[0], -30 * S)
    fig.text(0.01, (H + 0.05 + hb / 2) / height, "Highlighting \u2018Texas\u2019: darker = the J-lens ranks it higher",
             ha="left", va="center", fontsize=6.5, color=MUTED)
    out = HERE / "teaser.pdf"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out} ({width:.2f} x {height:.2f} in)")


def viewer_demo():
    """Fig. 4: viewer screenshots of a J-lens swap that works (undue -> rural) and one that fails (comfortable -> unfold).

    The panels are light-mode crops of the viewer on Gemma-3-4B-it, in viewer_demo/.
    """
    src = HERE / "viewer_demo"
    # One column per prompt: what the J-lens reads at the word, then the swap's result. The crops share one pixel
    # scale, so equal column widths keep the viewer's text the same size in every panel.
    columns = [("undue → rural: the swap works", "a_undue_reads_layer18", "b_undue_swap_result"),
               ("comfortable → unfold: the swap fails", "c_comfortable_reads_layer12", "d_comfortable_swap_result")]
    imgs = [[plt.imread(src / f"{p}.png") for p in col[1:]] for col in columns]
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[463, 217], wspace=0.06, hspace=0.05,
                          left=0.03, right=0.995, top=0.93, bottom=0.005)
    for j, (title, *_) in enumerate(columns):
        for i, img in enumerate(imgs[j]):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, interpolation="lanczos")
            ax.set_axis_off()
            ax.set_anchor("N")
            ax.text(-0.015, 0.98, f"({'acbd'[2 * j + i]})", transform=ax.transAxes, ha="right", va="top",
                    color=INK, fontsize=8, fontweight="bold")
            if i == 0:
                ax.set_title(title, color=INK, fontsize=8.5, pad=4)
    out = HERE / "viewer_demo.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


ANTONYM_BLOCKS = [("undue → rural: the swap works", "a_undue_reads_layer18", "b_undue_swap_result"),
                  ("comfortable → unfold: the swap fails", "c_comfortable_reads_layer12", "d_comfortable_swap_result")]
TWOHOP_BLOCKS = [("Texas → California at Dallas (layer 21): works", "twohop_a_dallas_reads_layer21",
                  "twohop_b_dallas_swap_result"),
                 ("Texas → California at the last token (layer 25): fails", "twohop_c_last_reads_layer25",
                  "twohop_d_last_swap_result")]


def viewer_demo_column(blocks=ANTONYM_BLOCKS, name="viewer_demo_column"):
    """Fig. 4 for one column: the same screenshots, cropped to the lens's top three words and the before/after lines.

    Each prompt gets its J-lens reading, (a) and (c), above the swap's result, (b) and (d).
    """
    src = HERE / "viewer_demo"
    card, result = (12, 22, 755, 292), (12, 120, 700, 205)  # (left, top, right, bottom) pixel crops
    crop = lambda name, box: plt.imread(src / f"{name}.png")[box[1]:box[3], box[0]:box[2]]
    imgs = [(title, crop(a, card), crop(b, result)) for title, a, b in blocks]
    ratio = lambda im: im.shape[0] / im.shape[1]
    fig = plt.figure(figsize=(3.4, 3.65))
    outer = fig.add_gridspec(2, 1, hspace=0.16, left=0.075, right=0.995, top=0.94, bottom=0.005)
    for k, (title, *pair) in enumerate(imgs):
        inner = outer[k].subgridspec(2, 1, height_ratios=[ratio(im) for im in pair], hspace=0.06)
        for i, img in enumerate(pair):
            ax = fig.add_subplot(inner[i])
            ax.imshow(img, interpolation="lanczos")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True), s.set_color("#d0d0d0"), s.set_linewidth(0.6)
            ax.text(-0.012, 0.98, f"({'abcd'[2 * k + i]})", transform=ax.transAxes, ha="right", va="top",
                    color=INK, fontsize=7.5, fontweight="bold")
            if i == 0:
                ax.set_title(title, color=INK, fontsize=8, pad=3, loc="left")
    out = HERE / f"{name}.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Wrote {out}")


def handoff(first_layer=4, thr=0.04):
    """Fig. 5 (one column): tracing the output back, and the copy edit at each layer, for "Q: Russia / A:".

    Left: the last four tokens' cells, labeled with the J-lens's top word and shaded by how much each feeds the
    output (the viewer's "Trace back"; outlined where it shows a cell, weight >= ``thr``). Right: the answer after
    copying the Russia cell, at that layer, from the same prompt ending in "Q: Japan". Data from
    viewer_demo/export_handoff.py and the viewer's edit endpoint.
    """
    src = HERE / "viewer_demo"
    d = json.loads((src / "handoff_trace_russia.json").read_text())
    patch = {int(l): w.strip() for l, (w, _) in json.loads((src / "handoff_patch_russia.json").read_text()).items()}
    rows = [i for i, l in enumerate(d["layers"]) if l >= first_layer][::-1]  # output layer on top, as in the viewer
    flow, sel = "#1a857e", "#c0392b"
    word = lambda w: (w.replace("\n", "⏎").strip() or "␣")[:9]
    head = [word(t) if t.strip() else "⏎" for t in d["tokens"]]

    fig, ax = plt.subplots(figsize=(3.4, 3.35))
    cw, ch, gap = 1.0, 1.0, 0.12
    xs = [0, 1, 2, 3, 4.35]
    for r, i in enumerate(rows):
        l, y = d["layers"][i], -r * ch
        for c in range(4):
            b = d["back"][i][c]
            ax.add_patch(plt.Rectangle((xs[c] + gap / 2, y + gap / 2), cw - gap, ch - gap, lw=0.7 if b >= thr else 0,
                                       edgecolor=flow, facecolor=flow, alpha=None))
            ax.patches[-1].set_facecolor((*matplotlib.colors.to_rgb(flow), min(1, b ** 0.5)))
            dark = b ** 0.5 > 0.55
            ax.text(xs[c] + cw / 2, y + ch / 2, word(d["words"][i][c]), ha="center", va="center", fontsize=4.6,
                    family="monospace", color="white" if dark else INK)
        ok = patch[l] == "Tokyo"
        ax.add_patch(plt.Rectangle((xs[4] + gap / 2, y + gap / 2), 1.3 - gap, ch - gap, lw=0,
                                   facecolor=JL if ok else "#e6e6e6"))
        ax.text(xs[4] + 0.65, y + ch / 2, patch[l], ha="center", va="center", fontsize=4.8, family="monospace",
                color="white" if ok else MUTED)
        ax.text(-0.15, y + ch / 2, "out" if l == d["layers"][-1] else str(l), ha="right", va="center", fontsize=5,
                color=MUTED)
    top = ch + 0.15
    for c, t in enumerate(head):
        ax.text(xs[c] + cw / 2, top, t, ha="center", va="bottom", fontsize=6, family="monospace", color=INK)
    ax.text(xs[4] + 0.65, top, "Russia ← Japan", ha="center", va="bottom", fontsize=5.5, color=INK)
    ax.text(1.95, top + 1.2, "J-lens reading, shaded by trace back", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.text(xs[4] + 0.65, top + 1.2, "answer after", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.add_patch(plt.Rectangle((xs[3] + gap / 2 - 0.05, gap / 2 - 0.05), cw - gap + 0.1, ch - gap + 0.1, lw=0.9,
                               edgecolor=sel, facecolor="none"))
    # The handoff: below the lowest layer where the copy no longer changes the answer.
    last_fail = max(r for r, i in enumerate(rows) if patch[d["layers"][i]] != "Tokyo")
    y_line = -last_fail * ch
    ax.plot([-0.9, xs[4] + 1.3], [y_line, y_line], color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(xs[4] + 1.38, y_line, "handoff", ha="left", va="center", fontsize=5.5, color=MUTED, rotation=90)
    ax.set_xlim(-1.0, xs[4] + 1.75)
    ax.set_ylim(-(len(rows) - 1) * ch - 0.1, top + 2.6)
    ax.set_aspect("auto")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    out = HERE / "handoff.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Wrote {out}")


def twohop_reading(cols=(7, 8, 9, 10), k=5):
    """One column: the Dallas question's last tokens over every layer, as the viewer's "Highlight a word" shows it.

    Cells are labeled with the J-lens's top word; shaded where Texas is among the top ``k`` (darker for a higher
    rank) and outlined where Austin is. Data from viewer_demo/export_twohop_reading.py.
    """
    d = json.loads((HERE / "viewer_demo" / "twohop_reading_dallas.json").read_text())
    word = lambda w: (w.replace("\n", "⏎").strip() or "␣")[:10]
    rows = list(range(len(d["layers"])))[::-1]  # output layer on top, as in the viewer
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    gap = 0.12
    for r, i in enumerate(rows):
        y = -r
        for c, p in enumerate(cols):
            texas, austin = d["ranks"][i][p]
            shade = (1.15 - texas / k) if texas <= k else 0
            ax.add_patch(plt.Rectangle((c + gap / 2, y + gap / 2), 1 - gap, 1 - gap, lw=0,
                                       facecolor=(*matplotlib.colors.to_rgb(JL), shade) if shade else "#f2f2f2"))
            if austin <= k:
                ax.add_patch(plt.Rectangle((c + gap / 2, y + gap / 2), 1 - gap, 1 - gap, lw=1.0, edgecolor=INK,
                                           facecolor="none"))
            ax.text(c + 0.5, y + 0.5, word(d["words"][i][p]), ha="center", va="center", fontsize=4.8,
                    family="monospace", color="white" if shade > 0.6 else INK)
        l = d["layers"][i]
        ax.text(-0.15, y + 0.5, "out" if l == d["layers"][-1] else str(l), ha="right", va="center", fontsize=5,
                color=MUTED)
    for c, p in enumerate(cols):
        ax.text(c + 0.5, 1.15, word(d["tokens"][p]), ha="center", va="bottom", fontsize=6.5, family="monospace",
                color=INK)
    ly = -len(rows) - 0.6  # legend under the grid
    ax.add_patch(plt.Rectangle((0.0, ly), 0.35, 0.8, facecolor=JL, lw=0))
    ax.text(0.45, ly + 0.4, f"Texas in the J-lens's top {k} (darker: higher rank)", va="center", fontsize=5.8,
            color=INK)
    ax.add_patch(plt.Rectangle((0.0, ly - 1.4), 0.35, 0.8, facecolor="none", edgecolor=INK, lw=1.0))
    ax.text(0.45, ly - 1.0, f"Austin in the top {k}", va="center", fontsize=5.8, color=INK)
    ax.set_xlim(-0.9, len(cols) + 0.05)
    ax.set_ylim(ly - 1.7, 2.2)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    out = HERE / "twohop_reading.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Wrote {out}")


COT_TASKS = [("retrieval", "Retrieval"), ("multihop", "Multi-hop"), ("serial", "Serial"), ("ordering", "Ordering")]
# Pattern tasks (antonyms, past tenses) are left out: their distractor is the top word more often than the answer.


def cot():
    """Fig. 6 (one column): is the final answer readable before the model writes any reasoning, per task type.

    Correct answers only, from experiments/cot/results/per_question.csv (100 questions per task type). (a) The answer
    is the lens's top word at some position before the reasoning, against a plausible wrong answer (the control).
    (b) The median first layer where it is, with the number of answers it is over.
    """
    import csv
    rows = [r for r in csv.DictReader((EXP / "cot" / "results" / "per_question.csv").open()) if r["correct"] == "True"]
    yes = lambda v: v == "True"
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.0), gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.75))
    x = np.arange(len(COT_TASKS))
    for j, (lens, color, name) in enumerate((("LL", LL, "logit lens"), ("JL", JL, "J-lens"))):
        ans = [np.mean([yes(r[f"{lens}_readable_top1"]) for r in rows if r["regime"] == t]) for t, _ in COT_TASKS]
        ctl = [np.mean([yes(r[f"{lens}_distractor_top1"]) for r in rows if r["regime"] == t]) for t, _ in COT_TASKS]
        a.bar(x + (j - 0.5) * 0.36, ans, 0.34, color=color, label=f"answer ({name})")
        a.scatter(x + (j - 0.5) * 0.36, ctl, s=14, marker="_", color=INK, linewidths=1.2, zorder=3,
                  label="control (wrong answer)" if j else None)
        med = [np.median([int(r[f"{lens}_first_layer"]) for r in rows if r["regime"] == t and r[f"{lens}_first_layer"]])
               for t, _ in COT_TASKS]
        cnt = [sum(1 for r in rows if r["regime"] == t and r[f"{lens}_first_layer"]) for t, _ in COT_TASKS]
        b.scatter(x + (j - 0.5) * 0.22, med, s=16, color=color, zorder=3, label=name)
        for xi, m, c in zip(x + (j - 0.5) * 0.22, med, cnt):
            b.text(xi + (0.06 if j else -0.06), m, f"{c}", fontsize=5.5, color=MUTED, va="center",
                   ha="left" if j else "right")
    n = {t: sum(1 for r in rows if r["regime"] == t) for t, _ in COT_TASKS}
    for ax in (a, b):
        ax.set_xticks(x, [f"{lab}\n(n={n[t]})" for t, lab in COT_TASKS] if ax is a else [lab for _, lab in COT_TASKS])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
    a.set_ylim(0, 1)
    a.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    a.set_ylabel("top word before\nthe reasoning")
    a.set_title("(a) Answer readable before any reasoning", loc="left", color=INK, pad=4)
    hs, ls = a.get_legend_handles_labels()
    order = [ls.index(l) for l in ("answer (logit lens)", "answer (J-lens)", "control (wrong answer)")]
    a.legend([hs[i] for i in order], [ls[i] for i in order], frameon=False, fontsize=6, ncol=3, loc="lower center",
             bbox_to_anchor=(0.5, 1.12), handlelength=1.2, columnspacing=1.0)
    b.set_ylim(14, 35)
    b.set_yticks([15, 20, 25, 30, 34])
    b.set_ylabel("median first layer")
    b.set_title("(b) How early, when it is (number of answers beside each)", loc="left", color=INK, pad=4)
    out = HERE / "cot.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    directions()
    twohop()
    twohop(column=True)
    twohop_mechanism()
    cot_schematic()
    edits_diagram()
    teaser()
    viewer_demo()
    viewer_demo_column()
    viewer_demo_column(TWOHOP_BLOCKS, "twohop_viewer")
    handoff()
    twohop_reading()
