"""Is the answer already readable at the entity token?

Swaps at the entity token work for country--capital and product--company but not
for antonym or English--French, although patching the entity cell works for all
four. One explanation: for fact lookups the model computes the answer *on the
entity token* (Geva et al., 2023), so later tokens read a ready answer, while for
antonyms and translation the few-shot task runs at the answer position and reads
the word's whole identity from the entity token.

This reads every layer of the entity token ("Q: France") with both lenses and
records the rank of this item's answer (" Paris"), the entity itself (" France")
and a control answer from another item. If the explanation holds, the answer is
readable at the entity token for fact lookups and not for antonym or translation.

    python experiments/edit_directions/answer_at_entity.py google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import residual_stream_vis as rsv
from residual_stream_vis.edit import _run

from run import FV_TASKS, HERE, build_items


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--tasks", default=",".join(FV_TASKS + ["legs"]))
    p.add_argument("--items", type=int, default=60, help="correctly answered items per task")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model)
    lens = rsv.JacobianLens.from_pretrained(a.model)
    out = Path(a.out or HERE / "results" / f"answer_at_entity-{a.model.replace('/', '__')}.jsonl")
    f = out.open("w")
    summary = {}
    for task in a.tasks.split(","):
        rng = random.Random(f"{a.seed}-{task}")
        items = build_items(tok, task, 5, rng)
        rng.shuffle(items)
        good = []
        for it in items:
            if len(good) >= a.items:
                break
            h, probs, ids = _run(model, tok, it["prompt"])
            pos = max((j for j, t in enumerate(ids) if t == it["entity_id"]), default=None)
            if int(probs.argmax()) == it["answer_id"] and pos is not None:
                good.append(dict(it, h=h, pos=pos))
        frac = defaultdict(list)
        for it in good:
            ctrl = rng.choice([o for o in good if o["answer_id"] != it["answer_id"]])
            watch = [it["answer_id"], it["entity_id"], ctrl["answer_id"]]
            for layer in lens.layers:
                for kind, L in (("logit", None), ("jlens", lens)):
                    logits = rsv.lens_logits(model, L, it["h"][layer, it["pos"]][None], layer, on_device=True)[0]
                    ranks = [int((logits > logits[i]).sum()) for i in watch]
                    f.write(json.dumps(dict(model=a.model, task=task, item=it["x"], layer=layer, lens=kind,
                                            answer=ranks[0], entity=ranks[1], ctrl_answer=ranks[2])) + "\n")
                    for w, r in zip(("answer", "entity", "ctrl_answer"), ranks):
                        frac[kind, w, layer].append(r < a.k)
        summary[task] = (len(good), frac)
        f.flush()
        print(f"[{task}] {len(good)} items", flush=True)
    f.close()

    layers = [l for l in lens.layers if l < lens.layers[-1]]
    print(f"\nFraction of items with the word in the lens's top {a.k} at the entity token "
          f"(max over layers, and at layers 12 / 20)")
    for task, (n, frac) in summary.items():
        for kind in ("jlens", "logit"):
            cells = []
            for w in ("answer", "entity", "ctrl_answer"):
                r = {l: np.mean(frac[kind, w, l]) for l in layers}
                best = max(r, key=r.get)
                cells.append(f"{w} {r[best]:.0%} (L{best}) {r.get(12, np.nan):.0%}/{r.get(20, np.nan):.0%}")
            print(f"  {task:16} n={n:3} {kind:5}  " + "   ".join(cells))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
