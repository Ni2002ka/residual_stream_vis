"""Does the J-lens read a word at its own token because the word predicts itself later?

The J-lens reads country and product names at their own token from layer 1, but
antonym and translation words only from about layer 16. The J-lens matrix is the
average effect of a cell on the final predictions at its own and later positions,
over ordinary text. So a hypothesis: the J-lens sees a word in its own cell when
having that word in the text makes the model expect the same word later (names
recur in a document; most adjectives do not).

For each word (25 query words from each of 5 tasks), two numbers:

* **self-expectation**: put the word in a short frame ("The word of the day is
  {w}.") followed by a fixed neutral passage, and average the word's log
  probability over the passage's positions. Subtract the same average when the
  frame holds one of the other words instead. This is how much seeing the word
  raises the model's expectation of it later.
* **J-lens readability**: the J-lens rank of the word at its own token in "He said
  {w}", at layers 1, 4, 8 and 12.

If the hypothesis holds, the two correlate, also within a task.

    python experiments/edit_directions/self_expectation.py google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

import residual_stream_vis as rsv
from residual_stream_vis.edit import _run

from run import HERE, build_items

FRAMES = ["The word of the day is{w}.", "Notes from the meeting:{w}."]
PASSAGE = ("\n\nThe meeting ended late in the afternoon. Everyone went home, and the next morning the report was "
           "sent to the whole team. It described what had happened, who had been there, and what would come next.")
LAYERS = [1, 4, 8, 12]


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--tasks", default="country-capital,product-company,legs,antonym,english-french")
    p.add_argument("--words", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model)
    lens = rsv.JacobianLens.from_pretrained(a.model)
    device = next(model.parameters()).device
    words = []  # (word, token id, task)
    for task in a.tasks.split(","):
        items = build_items(tok, task, 5, random.Random(f"self-{a.seed}-{task}"))
        random.Random(a.seed).shuffle(items)
        seen = {}
        for it in items:
            seen.setdefault(it["x"], it["entity_id"])
        words += [(w, i, task) for w, i in list(seen.items())[:a.words]]
    ids = torch.tensor([i for _, i, _ in words])

    # logp[f, v, w]: mean log p(word w) over the passage when frame f holds word v.
    logp = torch.zeros(len(FRAMES), len(words), len(words))
    n_passage = len(tok(PASSAGE, add_special_tokens=False)["input_ids"])
    for f, frame in enumerate(FRAMES):
        for v, (w, _, _) in enumerate(words):
            enc = tok(frame.format(w=" " + w) + PASSAGE, return_tensors="pt")
            lp = model(**{k: t.to(device) for k, t in enc.items()}).logits[0].float().log_softmax(-1).cpu()
            # The prediction at position j is for token j + 1; score the passage's tokens after the first.
            logp[f, v] = lp[-n_passage:-1][:, ids].mean(0)
        print(f"  frame {f} done", flush=True)
    P = logp.mean(0)
    off = ~torch.eye(len(words), dtype=bool)
    expect = (P.diag() - (P * off).sum(0) / off.sum(0)).numpy()

    ranks = {layer: [] for layer in LAYERS}
    for w, i, _ in words:
        h, _, _ = _run(model, tok, f"He said {w}")
        for layer in LAYERS:
            s = rsv.lens_logits(model, lens, h[layer, -1][None], layer, on_device=True)[0]
            ranks[layer].append(int((s > s[i]).sum()))

    rows = [dict(word=w, task=t, self_expectation=float(e), **{f"jlens_rank_L{l}": ranks[l][k] for l in LAYERS})
            for k, ((w, _, t), e) in enumerate(zip(words, expect))]
    out = HERE / "results" / f"self_expectation-{a.model.replace('/', '__')}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    tasks = a.tasks.split(",")
    print("\nPer task: median self-expectation (nats) and median J-lens rank of the word at its own token")
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        print(f"  {t:16} expect {np.median([r['self_expectation'] for r in rs]):5.2f}   "
              + "  ".join(f"L{l} {np.median([r[f'jlens_rank_L{l}'] for r in rs]):6.0f}" for l in LAYERS))
    print("\nSpearman(self-expectation, -J-lens rank): pooled, then within each task")
    for l in LAYERS:
        cells = [f"pooled {spearman([r['self_expectation'] for r in rows], [-r[f'jlens_rank_L{l}'] for r in rows]):5.2f}"]
        for t in tasks:
            rs = [r for r in rows if r["task"] == t]
            cells.append(f"{t} {spearman([r['self_expectation'] for r in rs], [-r[f'jlens_rank_L{l}'] for r in rs]):5.2f}")
        print(f"  L{l:2}  " + "  ".join(cells))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
