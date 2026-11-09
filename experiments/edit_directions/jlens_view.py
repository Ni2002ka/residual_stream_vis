"""How does the J-lens see the change a working swap makes?

The J-lens swap of *France* for *Japan* at the entity token moves the cell along
the J-lens directions of the two words. It works for countries but not for
antonyms, and copying the whole cell from the *Japan* prompt works for both. So
take the change copying makes (the needed change, ``h(Japan) - h(France)`` at the
entity token) and read it with the J-lens: how much do the J-lens scores of each
word move, and where do the new and old words rank among all words?

If the J-lens sees the needed change as "+Japan, -France", the J-lens swap points
the right way. If it sees "+tall-related words" or nothing word-specific, a swap
along two word directions cannot reproduce it.

The same pairs are also read in a neutral sentence ("He said {w}"), to tell a
property of the words from a property of the task.

Pairs are chosen as in ``word_swaps.py``, so rows join with its results.

    python experiments/edit_directions/jlens_view.py google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import residual_stream_vis as rsv
from residual_stream_vis.edit import _run

from diagnose import counterfactual
from run import HERE, Directions, build_items

LAYERS = [4, 8, 12, 16, 20]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--tasks", default="country-capital,product-company,antonym,english-french,legs")
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model)
    lens = rsv.JacobianLens.from_pretrained(a.model)
    dirs = Directions(model, lens)
    out = Path(HERE / "results" / f"jlens_view-{a.model.replace('/', '__')}.jsonl")
    f = out.open("w")
    agg = defaultdict(list)
    examples = defaultdict(list)

    def read(h, layer):
        return rsv.lens_logits(model, lens, h[None], layer, on_device=True)[0].float().cpu()

    for task in a.tasks.split(","):
        rng = random.Random(f"words-{a.seed}-{task}")
        items = build_items(tok, task, 5, rng)
        rng.shuffle(items)
        good = []
        for it in items:
            if len(good) >= 60:
                break
            h, probs, ids = _run(model, tok, it["prompt"])
            pos = max((j for j, t in enumerate(ids) if t == it["entity_id"]), default=None)
            if int(probs.argmax()) == it["answer_id"] and pos is not None:
                good.append(dict(it, h=h, entity_pos=pos))
        pairs = [(o, n) for o in good for n in good if o["answer_id"] != n["answer_id"]]

        for old, new in rng.sample(pairs, min(a.pairs, len(pairs))):
            h_cf, _, _ = _run(model, tok, counterfactual(old, new))
            pos = old["entity_pos"]
            g_old = _run(model, tok, f"He said {old['x']}")[0][:, -1]
            g_new = _run(model, tok, f"He said {new['x']}")[0][:, -1]
            for where, before, after in (("task", old["h"][:, pos], h_cf[:, pos]), ("generic", g_old, g_new)):
                for layer in LAYERS:
                    s0, s1 = read(before[layer], layer), read(after[layer], layer)
                    change = s1 - s0
                    rank_new = int((change > change[new["entity_id"]]).sum())
                    rank_old = int((-change > -change[old["entity_id"]]).sum())
                    self_old = int((s0 > s0[old["entity_id"]]).sum())
                    need = after[layer] - before[layer]
                    g = (dirs(new["entity_id"], layer, "jlens", False) - dirs(old["entity_id"], layer, "jlens", False))
                    row = dict(task=task, where=where, old=old["x"], new=new["x"], layer=layer,
                               rank_new_up=rank_new, rank_old_down=rank_old, self_rank_old=self_old,
                               cos=float(torch.cosine_similarity(need, g, 0)),
                               top_up=[tok.decode(i) for i in change.topk(8).indices],
                               top_down=[tok.decode(i) for i in (-change).topk(8).indices],
                               top_before=[tok.decode(i) for i in s0.topk(8).indices])
                    f.write(json.dumps(row) + "\n")
                    agg[task, where, layer].append(row)
                    if where == "task" and layer == 12 and len(examples[task]) < 3:
                        examples[task].append(row)
        f.flush()
        print(f"[{task}] done", flush=True)
    f.close()

    print("\nThe needed change (copying the cell from the new-word prompt), read with the J-lens.")
    print("Share of pairs where the new word is among the 5 words the change raises most / the old word among the 5 it")
    print("lowers most; median ranks; cosine between the needed change and the J-lens swap's direction.")
    for task in a.tasks.split(","):
        for where in ("task", "generic"):
            cells = []
            for layer in LAYERS:
                rs = agg[task, where, layer]
                up = np.mean([r["rank_new_up"] < 5 for r in rs])
                down = np.mean([r["rank_old_down"] < 5 for r in rs])
                cells.append(f"L{layer} {up:.0%}/{down:.0%} (med {np.median([r['rank_new_up'] for r in rs]):.0f}) "
                             f"cos {np.mean([r['cos'] for r in rs]):.2f}")
            print(f"  {task:16} {where:7} " + "  ".join(cells))
    print("\nExamples at layer 12 (task prompt): what the J-lens reads before, and what the needed change raises / lowers")
    for task, rows in examples.items():
        for r in rows:
            print(f"  {task:16} {r['old']}→{r['new']}\n      before: {r['top_before'][:6]}\n"
                  f"      up:     {r['top_up'][:6]}\n      down:   {r['top_down'][:6]}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
