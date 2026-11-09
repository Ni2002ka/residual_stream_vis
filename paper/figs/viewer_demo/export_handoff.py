"""Data for the handoff figure: the viewer's J-lens readouts and trace-back weights for the last four tokens.

Uses the viewer's own functions (``record``, ``mixing``, ``flow_weights``, ``read_cells``), on the country--capital
prompt that ends in "Q: Russia\\nA:". The copy-edit results per layer, ``handoff_patch_russia.json``, come from the
viewer's /api/edit endpoint (copy the Russia cell from the same prompt ending in "Q: Japan").

    .venv/bin/python paper/figs/viewer_demo/export_handoff.py
"""

import json
from pathlib import Path

import residual_stream_vis as rsv
from residual_stream_vis.flow import flow_weights, mixing, read_cells, record

SHOTS = ("Q: Madagascar\nA: Antananarivo\n\nQ: Congo\nA: Kinshasa\n\nQ: Philippines\nA: Manila\n\n"
         "Q: Botswana\nA: Gaborone\n\nQ: Thailand\nA: Bangkok\n\n")
PROMPT = SHOTS + "Q: Russia\nA:"

model, tok = rsv.load_model("google/gemma-3-4b-it")
lens = rsv.JacobianLens.from_pretrained("google/gemma-3-4b-it")
rec = record(model, tok, PROMPT)
n_layers, n = rec["h"].shape[:2]
stay, move = mixing(rec)
w = flow_weights(stay, move, n_layers - 1, n - 1, direction="back")
layers = list(range(1, n_layers))
probs, ids, _ = read_cells(model, lens, rec["h"], layers, k=1)
cols = list(range(n - 4, n))
out = dict(prompt=PROMPT, tokens=[tok.decode(rec["ids"][p]) for p in cols], layers=layers,
           words=[[tok.decode(int(ids[i, p, 0])) for p in cols] for i in range(len(layers))],
           probs=[[round(float(probs[i, p, 0]), 3) for p in cols] for i in range(len(layers))],
           back=[[round(float(w[l, p]), 4) for p in cols] for l in layers],
           back_bos=[round(float(w[l, 0]), 4) for l in layers])
path = Path(__file__).with_name("handoff_trace_russia.json")
path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"Wrote {path}")
