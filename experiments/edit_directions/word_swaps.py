"""Why do word swaps fail? Try directions taken from other prompts instead of the lens.

On antonym and English--French, copying the entity cell from a prompt naming the
new word works (80--88%), but swapping the lens's word direction does not. So the
cell carries the word's identity, just not along the lens's direction. Per pair
(old, new) and layer, at the entity token:

* ``jlens``: the J-lens swap from run.py (strength 2), for reference.
* ``mean_task``: add ``mean(new) - mean(old)``, where ``mean(w)`` is the entity
  cell averaged over 4 other prompts of the same task whose query is ``w``, with
  different shots. If this works, the identity is linear and holds across contexts.
* ``mean_generic``: the same, with ``w`` at the end of 4 generic sentences
  ("He said big"). If this works too, a tool could precompute the direction.
* ``patch``: the entity cell copied from this prompt with the new word (ceiling).

Size controls (``--kinds``), since the mean differences are 3--10x larger edits:

* ``jlens_sized`` / ``unembed_sized``: the lens swap's change, rescaled to the
  size of ``mean_generic``'s.
* ``random_sized``: a random direction of that size.
* ``generic_small``: ``mean_generic``'s change, shrunk to the J-lens swap's size.

Without averaging (``--kinds``):

* ``one_sentence``: the difference in one generic sentence ("He said big").
* ``bare``: the difference with the word alone after BOS.

    python experiments/edit_directions/word_swaps.py google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import residual_stream_vis as rsv
from residual_stream_vis.edit import _run, swap_vector

from diagnose import counterfactual, run_patches
from run import HERE, Directions, build_items, random_unit

GENERIC = ["He said {w}", "The next word is {w}", "I wrote down {w}", "She looked up {w}"]


def last_cells(model, tok, prompts: list[str]) -> torch.Tensor:
    """Mean residual stream at the last token of each prompt: ``(layers + 1, d)``."""
    return torch.stack([_run(model, tok, p)[0][:, -1] for p in prompts]).mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--tasks", default="antonym,english-french,country-capital,legs")
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--contexts", type=int, default=4, help="other prompts averaged per word")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--kinds", default="jlens,mean_task,mean_generic,patch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model)
    lens = rsv.JacobianLens.from_pretrained(a.model)
    device = next(model.parameters()).device
    layers = lens.layers[:-1]
    dirs = Directions(model, lens)
    kinds = a.kinds.split(",")
    out = Path(a.out or HERE / "results" / f"word_swaps-{a.model.replace('/', '__')}.jsonl")
    f = out.open("w")
    t0 = time.time()
    rates = defaultdict(lambda: defaultdict(list))

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
                good.append(dict(it, h=h, ids=ids, entity_pos=pos))
        pairs = [(o, n) for o in good for n in good if o["answer_id"] != n["answer_id"]]

        def task_mean(word: str, avoid: str) -> torch.Tensor:
            """The entity cell for ``word`` averaged over other prompts of this task (fresh shots, not ``avoid``'s)."""
            cells = []
            r = random.Random(f"ctx-{task}-{word}")
            while len(cells) < a.contexts:
                proto = r.choice(items)
                prompt = proto["prompt"]
                if prompt == avoid or proto["x"] == word:
                    continue
                # Rename the proto's query to ``word``; the query ends the prompt, so positions line up from the end.
                q = f"a {proto['x']} have?" if task == "legs" else f"Q: {proto['x']}\n"
                i = prompt.rindex(q)
                prompt = prompt[:i] + q.replace(proto["x"], word) + prompt[i + len(q):]
                h, _, ids = _run(model, tok, prompt)
                wid = tok(" " + word, add_special_tokens=False)["input_ids"][0]
                cells.append(h[:, max(j for j, t in enumerate(ids) if t == wid)])
            return torch.stack(cells).mean(0)

        for i, (old, new) in enumerate(rng.sample(pairs, min(a.pairs, len(pairs)))):
            h_cf, _, _ = _run(model, tok, counterfactual(old, new))
            mt = task_mean(new["x"], old["prompt"]) - task_mean(old["x"], old["prompt"]) if "mean_task" in kinds else None
            mg = (last_cells(model, tok, [g.format(w=new["x"]) for g in GENERIC])
                  - last_cells(model, tok, [g.format(w=old["x"]) for g in GENERIC]))
            one = {k: last_cells(model, tok, [p.format(w=new["x"])]) - last_cells(model, tok, [p.format(w=old["x"])])
                   for k, p in (("one_sentence", GENERIC[0]), ("bare", " {w}")) if k in kinds}
            pos = old["entity_pos"]
            enc = {k: v.to(device) for k, v in tok(old["prompt"], return_tensors="pt").items()}
            runs, patches = [], []
            for layer in layers:
                h = old["h"][layer, pos]
                swap = {k: swap_vector(h, dirs(old["entity_id"], layer, k, center=False),
                                       dirs(new["entity_id"], layer, k, center=False), 2.0) - h
                        for k in ("jlens", "unembed")}
                big = mg[layer].norm()
                unit = lambda v: v / v.norm().clamp_min(1e-12)
                edits = {
                    "jlens": lambda: h + swap["jlens"],
                    "mean_task": lambda: h + mt[layer],
                    "mean_generic": lambda: h + mg[layer],
                    "patch": lambda: h_cf[layer, pos],
                    "jlens_sized": lambda: h + big * unit(swap["jlens"]),
                    "unembed_sized": lambda: h + big * unit(swap["unembed"]),
                    "random_sized": lambda: h + big * random_unit(h.numel(), a.seed, task, i, layer),
                    "generic_small": lambda: h + swap["jlens"].norm() * unit(mg[layer]),
                    "one_sentence": lambda: h + one["one_sentence"][layer],
                    "bare": lambda: h + one["bare"][layer],
                }
                for kind in kinds:
                    c = edits[kind]()
                    runs.append(dict(layer=layer, kind=kind,
                                     size=float((c - h).norm() / h.norm()),
                                     cos_to_patch=float(torch.cosine_similarity(c - h, h_cf[layer, pos] - h, 0))))
                    patches.append((layer, [pos], c[None]))
            tops, pr = run_patches(model, enc, patches, [new["answer_id"], old["answer_id"]], a.batch)
            for r, top, (p_new, p_old) in zip(runs, tops, pr.tolist()):
                ok = top == new["answer_id"]
                rates[task, r["kind"]][r["layer"]].append((ok, r["cos_to_patch"], r["size"]))
                f.write(json.dumps(dict(model=a.model, task=task, old=old["x"], new=new["x"], **r, success=ok,
                                        kept=top == old["answer_id"], p_new=p_new, p_old=p_old)) + "\n")
            print(f"  [{task}] pair {i + 1}  {old['x']}→{new['x']}  {time.time() - t0:.0f}s", flush=True)
        f.flush()
    f.close()

    print("\nSuccess at the best layer (layer), and mean cosine to the patch change / size relative to the cell at layer 12")
    for task in a.tasks.split(","):
        cells = []
        for kind in kinds:
            d = rates[task, kind]
            if not d:
                continue
            s = {l: np.mean([x[0] for x in v]) for l, v in d.items()}
            b = max(s, key=s.get)
            c12 = np.mean([x[1] for x in d[12]]) if 12 in d else float("nan")
            z12 = np.mean([x[2] for x in d[12]]) if 12 in d else float("nan")
            cells.append(f"{kind} {s[b]:.0%} (L{b}) cos {c12:.2f} size {z12:.2f}")
        print(f"  {task:16} " + "   ".join(cells))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
