"""Cosine between the logit-lens and J-lens direction of the same word, per layer (Fig. 3c).

The words are the push targets of the four function-vector tasks in run.py's results
(the leg-count task is left out). Writes results/direction_cosine.json.
"""
import json

import torch

import residual_stream_vis as rsv
from run import Directions, one_token

MODEL = "google/gemma-3-4b-it"
TASKS = {"country-capital", "product-company", "antonym", "english-french"}

model, tok = rsv.load_model(MODEL)
lens = rsv.JacobianLens.from_pretrained(MODEL)
d = Directions(model, lens)

ids = set()
for line in open("results/google__gemma-3-4b-it.jsonl"):
    r = json.loads(line)
    if (r.get("edit") == "push" and r["task"] in TASKS and r["layer"] == 1
            and r["kind"] == "unembed" and r["strength"] == 1.0):
        ids.add(one_token(tok, r["target"])[0])
ids = sorted(ids)

out = {"model": MODEL, "tasks": sorted(TASKS), "n_words": len(ids), "centered": True, "layers": {}}
for layer in lens.layers:
    cs = torch.tensor([float(d(i, layer, "unembed", True) @ d(i, layer, "jlens", True)) for i in ids])
    out["layers"][layer] = dict(mean=float(cs.mean()), q25=float(cs.quantile(.25)), q75=float(cs.quantile(.75)))
json.dump(out, open("results/direction_cosine.json", "w"), indent=1)
print(len(ids), {l: round(v["mean"], 2) for l, v in out["layers"].items() if l in (8, 12, 20, 27, 33)})
