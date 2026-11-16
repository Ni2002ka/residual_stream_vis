"""Logit-lens vs J-lens edit directions, swept over layers, with a random control.

Two edits, each tried with three directions (``unembed``: the word's unembedding
row, the logit lens's direction; ``jlens``: that row pulled back through ``J``;
``random``: a random direction of the same size):

* **push** changes what the edited token outputs. At the last token of a
  few-shot prompt, add a word's direction scaled by ``alpha × |cell|``.
  Success: the target word becomes the top prediction.
* **swap** changes a concept that later tokens read. At the query's entity
  token ("Q: France"), swap the entity's direction for another's, as
  ``rsv.swap`` does. Success: the model gives the new entity's answer.

Tasks are function_vectors' input → output lists (Todd et al., 2024) as
"Q: x\\nA: y" few-shot prompts, plus the viewer's leg-count prompt. Only items
the model answers correctly, with one-token entities and answers, are used.

    python experiments/edit_directions/run.py google/gemma-3-1b-pt
    python experiments/edit_directions/run.py gpt2 --n 20 --layers 2,6,10   # quick look
    python experiments/edit_directions/analyze.py experiments/edit_directions/results/*.jsonl

Writes one JSON line per (task, edit, item, layer, kind, strength) to
``results/<model>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

import torch

import residual_stream_vis as rsv
from residual_stream_vis.blocks import _first_tensor
from residual_stream_vis.edit import _run, _swap_first, swap_vector

HERE = Path(__file__).parent
FV_COMMIT = "fb9eac7b6dc707ea1475a717379916007fe448d5"
FV_URL = f"https://raw.githubusercontent.com/ericwtodd/function_vectors/{FV_COMMIT}/dataset_files/abstractive/{{}}.json"
FV_TASKS = ["country-capital", "antonym", "english-french", "product-company"]

# The viewer's leg-count example (server.DEFAULT_EXAMPLES): three shots, then the question.
LEGS_SHOTS = ("Q: How many legs does a frog have?\nA: 4\n\nQ: How many legs does a crab have?\nA: 10\n\n"
              "Q: How many legs does a human have?\nA: 2\n\n")
LEGS = {
    "spider": 8, "scorpion": 8, "tick": 8,
    "ant": 6, "bee": 6, "beetle": 6, "fly": 6, "wasp": 6, "butterfly": 6, "cockroach": 6, "grasshopper": 6, "mosquito": 6,
    "dog": 4, "cat": 4, "horse": 4, "cow": 4, "pig": 4, "sheep": 4, "lion": 4, "tiger": 4, "elephant": 4,
    "mouse": 4, "rabbit": 4, "lizard": 4,
    "chicken": 2, "duck": 2, "owl": 2, "penguin": 2, "ostrich": 2, "eagle": 2, "parrot": 2,
    "snake": 0, "worm": 0,
}
PUSH_ALPHAS = [0.5, 1.0, 2.0]
SWAP_STRENGTHS = [1.0, 2.0, 4.0]


def one_token(tok, word: str) -> tuple[int, str] | None:
    """``(id, prompt ending)`` for the token a model writes after "A:", or None if ``word`` isn't one token.

    " Paris" follows "A:"; a digit that only exists without its space (Gemma's "8") follows "A: ".
    """
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return ids[0], "A:"
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return (ids[0], "A: ") if len(ids) == 1 else None


def load_fv(task: str) -> list[dict]:
    path = HERE / "data" / f"{task}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FV_URL.format(task)) as r:
            path.write_bytes(r.read())
    return [{"x": d["input"], "y": d["output"]} for d in json.loads(path.read_text())]


def build_items(tok, task: str, shots: int, rng: random.Random) -> list[dict]:
    """Prompts whose entity and answer are each one token. ``entity_id`` is the token to find in the prompt."""
    if task == "legs":
        raw = [{"x": a, "y": str(n)} for a, n in LEGS.items()]
    else:
        raw = load_fv(task)
    items = []
    for d in raw:
        ans, ent = one_token(tok, d["y"]), tok(" " + d["x"], add_special_tokens=False)["input_ids"]
        if ans is None or len(ent) != 1:
            continue
        if task == "legs":
            prompt = LEGS_SHOTS + f"Q: How many legs does a {d['x']} have?\n{ans[1]}"
        else:
            others = rng.sample([o for o in raw if o["x"] != d["x"]], shots)
            prompt = "".join(f"Q: {o['x']}\nA: {o['y']}\n\n" for o in others) + f"Q: {d['x']}\n{ans[1]}"
        items.append(dict(task=task, x=d["x"], y=d["y"], prompt=prompt, answer_id=ans[0], entity_id=ent[0]))
    return items


class Directions:
    """Unit word directions at a layer, as ``rsv.word_direction`` computes them, without copying ``W_U`` each call."""

    def __init__(self, model, lens):
        self.w = model.get_output_embeddings().weight.detach()
        self.mean = self.w.mean(0, dtype=torch.float32).cpu()
        self.lens = lens

    def __call__(self, token_id: int, layer: int, kind: str, center: bool) -> torch.Tensor:
        v = self.w[token_id].float().cpu()
        if center:
            v = v - self.mean
        if kind == "jlens":
            J = self.lens.matrix(layer)
            if J is not None:
                v = v @ J
        return v / v.norm().clamp_min(1e-12)


def random_unit(d: int, *key) -> torch.Tensor:
    """A random unit vector, the same for the same ``key`` in every run."""
    v = torch.randn(d, generator=torch.Generator().manual_seed(zlib.crc32(repr(key).encode())))
    return v / v.norm()


@torch.no_grad()
def next_probs(model, enc, edits) -> torch.Tensor:
    """One edited run's next-word probabilities (the slow, unbatched reference for :func:`run_cells`)."""
    with rsv.editing(model, edits):
        return model(**enc).logits[0, -1].float().softmax(-1).cpu()


@torch.no_grad()
def run_cells(model, enc, position: int, cells: list[tuple[int, torch.Tensor]], watch: list[int], batch: int):
    """Run the prompt once per ``(layer, new cell)``, batched, each row with its own cell replaced.

    Returns the top token of each run and the probabilities of the ``watch`` tokens, ``(len(cells), len(watch))``.
    """
    blocks = rsv.find_blocks(model)
    tops, probs = [], []
    for s in range(0, len(cells), batch):
        chunk = cells[s:s + batch]
        rows = defaultdict(list)
        for r, (layer, cell) in enumerate(chunk):
            rows[layer].append((r, cell))

        def post(m, args, out, rc):
            h = _first_tensor(out).clone()
            h[[r for r, _ in rc], position] = torch.stack([c for _, c in rc]).to(h.device, h.dtype)
            return _swap_first(out, h)

        handles = [blocks[layer - 1].register_forward_hook(lambda m, a, o, rc=rc: post(m, a, o, rc))
                   for layer, rc in rows.items()]
        try:
            logits = model(**{k: v.expand(len(chunk), -1) for k, v in enc.items()}).logits[:, -1].float()
        finally:
            for hd in handles:
                hd.remove()
        p = logits.softmax(-1)
        tops += p.argmax(-1).tolist()
        probs.append(p[:, watch].cpu())
    return tops, torch.cat(probs)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--lens", default=None, help="J-lens path (default: the published one for this model)")
    p.add_argument("--tasks", default=",".join(FV_TASKS + ["legs"]))
    p.add_argument("--edits", default="push,swap")
    p.add_argument("--n", type=int, default=100, help="max items (push) or pairs (swap) per task")
    p.add_argument("--max-items", type=int, default=200,
                   help="stop checking a task's items once this many are answered correctly")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--layers", default="all", help="'all' (every J-lens layer but the last) or e.g. 2,4,6")
    p.add_argument("--batch", type=int, default=32, help="edited runs per forward pass")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model, device=a.device)
    lens = rsv.JacobianLens.load(a.lens) if a.lens else rsv.JacobianLens.from_pretrained(a.model)
    device = next(model.parameters()).device
    # The top layer's J is the identity, so both kinds are the same there.
    layers = lens.layers[:-1] if a.layers == "all" else [int(l) for l in a.layers.split(",")]
    dirs = Directions(model, lens)
    out = Path(a.out or HERE / "results" / f"{a.model.replace('/', '__')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w")
    t0 = time.time()

    for task in a.tasks.split(","):
        rng = random.Random(f"{a.seed}-{task}")
        items = build_items(tok, task, a.shots, rng)
        n_one_token = len(items)
        rng.shuffle(items)
        checked = []
        for it in items:
            if sum(c["correct"] and c["entity_pos"] is not None for c in checked) >= a.max_items:
                break
            checked.append(it)
            it["enc"] = {k: v.to(device) for k, v in tok(it["prompt"], return_tensors="pt").items()}
            h, probs, ids = _run(model, tok, it["prompt"])
            it["h"], it["ids"] = h, ids
            it["correct"] = int(probs.argmax()) == it["answer_id"]
            it["p_answer"] = float(probs[it["answer_id"]])
            it["entity_pos"] = max((j for j, t in enumerate(ids) if t == it["entity_id"]), default=None)
        items = [it for it in checked if it["correct"] and it["entity_pos"] is not None]
        print(f"[{task}] {n_one_token} one-token items; checked {len(checked)}, {len(items)} answered correctly", flush=True)
        f.write(json.dumps(dict(meta=True, model=a.model, task=task, n_one_token=n_one_token, n_checked=len(checked),
                                n_correct=len(items))) + "\n")
        if len(items) < 2:
            continue

        def write(**r):
            f.write(json.dumps(dict(model=a.model, task=task, **r)) + "\n")

        if "push" in a.edits:
            for i, it in enumerate(items[:a.n]):
                target = rng.choice([o for o in items if o["answer_id"] != it["answer_id"]])
                pos = len(it["ids"]) - 1
                runs, cells = [], []
                for layer in layers:
                    h = it["h"][layer, pos]
                    for kind in ("unembed", "jlens", "random"):
                        d = (random_unit(h.numel(), a.seed, task, i, layer)
                             if kind == "random" else dirs(target["answer_id"], layer, kind, center=True))
                        for alpha in PUSH_ALPHAS:
                            runs.append(dict(layer=layer, kind=kind, strength=alpha))
                            cells.append((layer, h + alpha * h.norm() * d))
                tops, pr = run_cells(model, it["enc"], pos, cells, [target["answer_id"], it["answer_id"]], a.batch)
                for r, top, (p_target, p_answer) in zip(runs, tops, pr.tolist()):
                    write(edit="push", item=it["x"], target=target["y"], **r, success=top == target["answer_id"],
                          changed=top != it["answer_id"], p_target_after=p_target, p_answer_after=p_answer)
                print(f"  push {i + 1}/{min(a.n, len(items))}  {time.time() - t0:.0f}s", flush=True)

        if "swap" in a.edits:
            pairs = [(o, n) for o in items for n in items if o["answer_id"] != n["answer_id"]]
            for i, (old, new) in enumerate(rng.sample(pairs, min(a.n, len(pairs)))):
                pos = old["entity_pos"]
                runs, cells = [], []
                for layer in layers:
                    h = old["h"][layer, pos]
                    jl = {}  # the J-lens swap's change at each strength, to size the random control
                    for kind in ("unembed", "jlens", "random"):
                        for s in SWAP_STRENGTHS:
                            if kind == "random":  # a random change as big as the J-lens swap's
                                cell = h + jl[s].norm() * random_unit(h.numel(), a.seed, task, i, layer, s)
                            else:
                                cell = swap_vector(h, dirs(old["entity_id"], layer, kind, center=False),
                                                   dirs(new["entity_id"], layer, kind, center=False), s)
                                if kind == "jlens":
                                    jl[s] = cell - h
                            runs.append(dict(layer=layer, kind=kind, strength=s))
                            cells.append((layer, cell))
                tops, pr = run_cells(model, old["enc"], pos, cells, [new["answer_id"], old["answer_id"]], a.batch)
                for r, top, (p_new, p_old) in zip(runs, tops, pr.tolist()):
                    write(edit="swap", item=old["x"], new=new["x"], **r, success=top == new["answer_id"],
                          changed=top != old["answer_id"], p_new_after=p_new, p_old_after=p_old)
                print(f"  swap {i + 1}/{min(a.n, len(pairs))}  {time.time() - t0:.0f}s", flush=True)
        f.flush()
    f.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
