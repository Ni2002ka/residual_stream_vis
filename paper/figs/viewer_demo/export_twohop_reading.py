"""Data for the two-hop reading figure: the J-lens's top words and the ranks of Texas and Austin in every cell.

Uses the viewer's own functions (``record``, ``read_cells``) on "Fact: the capital of the state containing Dallas is".
Rank 1 is the lens's top word, as in the viewer's "Highlight a word".

    .venv/bin/python paper/figs/viewer_demo/export_twohop_reading.py
"""

import json
from pathlib import Path

import residual_stream_vis as rsv
from residual_stream_vis.flow import read_cells, record

PROMPT = "Fact: the capital of the state containing Dallas is"
TRACK = [" Texas", " Austin"]

model, tok = rsv.load_model("google/gemma-3-4b-it")
lens = rsv.JacobianLens.from_pretrained("google/gemma-3-4b-it")
rec = record(model, tok, PROMPT)
n_layers, n = rec["h"].shape[:2]
layers = list(range(1, n_layers))
track_ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in TRACK]
probs, ids, ranks = read_cells(model, lens, rec["h"], layers, k=1, track_ids=track_ids)
out = dict(prompt=PROMPT, tokens=[tok.decode(i) for i in rec["ids"]], layers=layers, track=TRACK,
           words=[[tok.decode(int(ids[i, p, 0])) for p in range(n)] for i in range(len(layers))],
           probs=[[round(float(probs[i, p, 0]), 3) for p in range(n)] for i in range(len(layers))],
           ranks=[[[int(r) for r in ranks[i, p]] for p in range(n)] for i in range(len(layers))])
path = Path(__file__).with_name("twohop_reading_dallas.json")
path.write_text(json.dumps(out, ensure_ascii=False))
print(f"Wrote {path}")
