"""Why do swaps fail? Compare them with activation patching from a prompt that names the new entity.

For each pair (old, new), the counterfactual prompt is the old prompt with the
query's entity replaced by the new one (same length: both are one token). Per layer:

* ``patch_entity``: the entity cell copied from the counterfactual run. The best any
  edit of that one cell can do.
* ``patch_after``: only the cells after the entity copied ("\\n", "A", ":" ...).
  High success here means the entity has already been read into later tokens.
* ``patch_both``: the entity cell and every cell after it.
* ``unembed`` / ``jlens``: the swap edits from run.py (strength 2), plus how big the
  removed component is (``|h·old| / |h|``) and how well the swap's change points
  at the patch's change (cosine).

    python experiments/edit_directions/diagnose.py google/gemma-3-4b-it --pairs 20
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

import residual_stream_vis as rsv
from residual_stream_vis.blocks import _first_tensor
from residual_stream_vis.edit import _run, _swap_first, swap_vector

from run import FV_TASKS, HERE, Directions, build_items


@torch.no_grad()
def run_patches(model, enc, patches: list[tuple[int, list[int], torch.Tensor]], watch: list[int], batch: int):
    """Like run.py's ``run_cells``, but each run replaces cells at several positions: ``(layer, positions, cells)``."""
    blocks = rsv.find_blocks(model)
    tops, probs = [], []
    for s in range(0, len(patches), batch):
        chunk = patches[s:s + batch]
        by_layer = {}
        for r, (layer, pos, cells) in enumerate(chunk):
            by_layer.setdefault(layer, []).append((r, pos, cells))

        def post(m, args, out, rows):
            h = _first_tensor(out).clone()
            for r, pos, cells in rows:
                h[r, pos] = cells.to(h.device, h.dtype)
            return _swap_first(out, h)

        handles = [blocks[layer - 1].register_forward_hook(lambda m, a, o, rows=rows: post(m, a, o, rows))
                   for layer, rows in by_layer.items()]
        try:
            logits = model(**{k: v.expand(len(chunk), -1) for k, v in enc.items()}).logits[:, -1].float()
        finally:
            for hd in handles:
                hd.remove()
        p = logits.softmax(-1)
        tops += p.argmax(-1).tolist()
        probs.append(p[:, watch].cpu())
    return tops, torch.cat(probs)


def counterfactual(old: dict, new: dict) -> str:
    """The old prompt with its query's entity renamed."""
    q = f"a {old['x']} have?" if old["task"] == "legs" else f"Q: {old['x']}\n"
    i = old["prompt"].rindex(q)
    return old["prompt"][:i] + q.replace(old["x"], new["x"]) + old["prompt"][i + len(q):]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--tasks", default=",".join(FV_TASKS + ["legs"]))
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--max-items", type=int, default=60)
    p.add_argument("--strength", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model)
    lens = rsv.JacobianLens.from_pretrained(a.model)
    device = next(model.parameters()).device
    layers = lens.layers[:-1]
    dirs = Directions(model, lens)
    out = Path(a.out or HERE / "results" / f"diagnose-{a.model.replace('/', '__')}.jsonl")
    f = out.open("w")
    t0, checked_once = time.time(), False

    for task in a.tasks.split(","):
        rng = random.Random(f"diag-{a.seed}-{task}")
        items = build_items(tok, task, 5, rng)
        rng.shuffle(items)
        good = []
        for it in items:
            if len(good) >= a.max_items:
                break
            h, probs, ids = _run(model, tok, it["prompt"])
            pos = max((j for j, t in enumerate(ids) if t == it["entity_id"]), default=None)
            if int(probs.argmax()) == it["answer_id"] and pos is not None:
                good.append(dict(it, h=h, ids=ids, entity_pos=pos))
        pairs = [(o, n) for o in good for n in good if o["answer_id"] != n["answer_id"]]
        for i, (old, new) in enumerate(rng.sample(pairs, min(a.pairs, len(pairs)))):
            cf = counterfactual(old, new)
            h_cf, p_cf, ids_cf = _run(model, tok, cf)
            pos, n = old["entity_pos"], len(old["ids"])
            assert len(ids_cf) == n and ids_cf[pos] == new["entity_id"], (old["x"], new["x"])
            base = dict(model=a.model, task=task, old=old["x"], new=new["x"], new_answer=new["y"], tokens_after=n - 1 - pos,
                        cf_top=tok.decode(int(p_cf.argmax())), cf_success=int(p_cf.argmax()) == new["answer_id"])
            enc = {k: v.to(device) for k, v in tok(old["prompt"], return_tensors="pt").items()}
            runs, patches = [], []
            after = list(range(pos + 1, n))
            for layer in layers:
                h, hc = old["h"][layer], h_cf[layer]
                runs += [dict(layer=layer, kind=k) for k in ("patch_entity", "patch_after", "patch_both")]
                patches += [(layer, [pos], hc[[pos]]), (layer, after, hc[after]), (layer, [pos] + after, hc[[pos] + after])]
                want = hc[pos] - h[pos]
                for kind in ("unembed", "jlens"):
                    u_old = dirs(old["entity_id"], layer, kind, center=False)
                    cell = swap_vector(h[pos], u_old, dirs(new["entity_id"], layer, kind, center=False), a.strength)
                    runs.append(dict(layer=layer, kind=kind, removed=float((h[pos] @ u_old).abs() / h[pos].norm()),
                                     cos_to_patch=float(torch.cosine_similarity(cell - h[pos], want, 0)),
                                     size_vs_patch=float((cell - h[pos]).norm() / want.norm())))
                    patches.append((layer, [pos], cell[None]))
            tops, pr = run_patches(model, enc, patches, [new["answer_id"], old["answer_id"]], a.batch)
            for r, top, (p_new, p_old) in zip(runs, tops, pr.tolist()):
                f.write(json.dumps(dict(**base, **r, success=top == new["answer_id"], changed=top != old["answer_id"],
                                        p_new_after=p_new, p_old_after=p_old)) + "\n")
            if not checked_once:  # the batched swap should match rsv.swap
                layer = layers[len(layers) // 2]
                ref = rsv.swap(model, tok, old["prompt"], old["x"], new["x"], layer, pos, lens=lens, kind="jlens",
                               strength=a.strength)
                mine = next(t for r, t in zip(runs, tops) if r["layer"] == layer and r["kind"] == "jlens")
                print(f"  check vs rsv.swap at layer {layer}: {tok.decode(mine)!r} vs {ref.top(1)[0][0]!r}", flush=True)
                checked_once = True
            print(f"[{task}] pair {i + 1}  {old['x']}→{new['x']}  {time.time() - t0:.0f}s", flush=True)
        f.flush()
    f.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
