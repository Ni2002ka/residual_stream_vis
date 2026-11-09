"""Is a cell "context part + word part"? The difference edit relies on it.

The one-sentence swap adds ``cell("He said fast") - cell("He said tall")`` to the
``tall`` cell of a task prompt. That is right if a cell at the word's token is
the sum of a part from the context and a part from the word: the context parts
cancel, and the word parts carry over to any context. This tests that directly.

Every word (10 query words from each of 4 tasks) goes into every context (the 4
generic sentences of ``word_swaps.py`` and one prompt from each task, its query
replaced). Per layer, the cells at the word's token are split as in a two-way
ANOVA, ``cell[c, w] = mean + context[c] + word[w] + interaction[c, w]``, and we
report each part's share of the variance. If the interaction is small, the sum
picture holds.

Also per layer, the transfer the edit actually makes: for two words of a task,
the cosine between their difference in a generic sentence and in that task's
prompt, and the error of the first as an estimate of the second.

    python experiments/edit_directions/additivity.py google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import itertools
import json
import random

import numpy as np
import torch

import residual_stream_vis as rsv
from residual_stream_vis.edit import _run

from run import HERE, LEGS_SHOTS, build_items
from word_swaps import GENERIC


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--tasks", default="antonym,english-french,country-capital,legs")
    p.add_argument("--words", type=int, default=10, help="query words per task")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model)
    tasks = a.tasks.split(",")
    words, contexts = [], {f"generic{i}": g for i, g in enumerate(GENERIC)}
    for task in tasks:
        items = build_items(tok, task, 5, random.Random(f"add-{a.seed}-{task}"))
        random.Random(a.seed).shuffle(items)
        seen = []
        for it in items:
            if it["x"] not in [w for w, _, _ in seen]:
                seen.append((it["x"], it["entity_id"], task))
        words += seen[:a.words]
        it = items[0]
        if task == "legs":
            contexts[task] = LEGS_SHOTS.replace("{", "{{").replace("}", "}}") + "Q: How many legs does a {w} have?\nA:"
        else:
            q = f"Q: {it['x']}\n"
            contexts[task] = it["prompt"][:it["prompt"].rindex(q)].replace("{", "{{").replace("}", "}}") + "Q: {w}\nA:"

    names = list(contexts)
    cells = None  # (contexts, words, layers + 1, d)
    for ci, c in enumerate(names):
        for wi, (w, wid, _) in enumerate(words):
            h, _, ids = _run(model, tok, contexts[c].format(w=w))
            pos = max(j for j, t in enumerate(ids) if t == wid)
            if cells is None:
                cells = torch.zeros(len(names), len(words), *h[:, pos].shape)
            cells[ci, wi] = h[:, pos]
        print(f"  context {c} done", flush=True)

    rows = []
    for layer in range(1, cells.shape[2]):
        X = cells[:, :, layer].double()
        share = {}
        # Raw, and with every dimension rescaled to unit spread so a few large dimensions can't dominate.
        for tag, Y in (("", X), ("std_", X / X.flatten(0, 1).std(0).clamp_min(1e-8))):
            mu = Y.mean((0, 1), keepdim=True)
            ctx, word = Y.mean(1, keepdim=True) - mu, Y.mean(0, keepdim=True) - mu
            total = ((Y - mu) ** 2).sum()
            share |= {tag + k: float((v.expand_as(Y) ** 2).sum() / total) for k, v in
                      (("context", ctx), ("word", word), ("interaction", Y - mu - ctx - word))}
        cos, err = [], []
        for task in tasks:
            idx = [i for i, (_, _, t) in enumerate(words) if t == task]
            for i, j in itertools.combinations(idx, 2):
                target = X[names.index(task), i] - X[names.index(task), j]
                for g in range(len(GENERIC)):
                    est = X[g, i] - X[g, j]
                    cos.append(float(torch.cosine_similarity(est, target, 0)))
                    err.append(float((est - target).norm() / target.norm()))
        rows.append(dict(layer=layer, **share, transfer_cos=float(np.mean(cos)), transfer_err=float(np.mean(err))))

    out = HERE / "results" / "additivity.json"
    out.write_text(json.dumps(dict(model=a.model, contexts=names, words=[w for w, _, _ in words], layers=rows), indent=1))
    print(f"\n{len(names)} contexts x {len(words)} words. Variance share of each part; generic -> task transfer")
    print("layer  context  word  interaction  (rescaled: context word interaction)   cos   rel. error")
    for r in rows:
        print(f"{r['layer']:5}  {r['context']:7.0%}  {r['word']:4.0%}  {r['interaction']:11.0%}"
              f"  ({r['std_context']:.0%} {r['std_word']:.0%} {r['std_interaction']:.0%})"
              f"   {r['transfer_cos']:.2f}  {r['transfer_err']:.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
