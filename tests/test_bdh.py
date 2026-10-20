import math

import pytest
import torch
import torch.nn.functional as F

import residual_stream_vis as rsv
from residual_stream_vis import bdh
from residual_stream_vis.flow import mixing, record


def tiny(seed=0):
    torch.manual_seed(seed)
    m = bdh.BDHForCausalLM(bdh.BDHConfig(n_layer=3, n_embd=16, n_head=2, mlp_internal_dim_multiplier=4, dropout=0.0))
    for p in m.parameters():  # bigger than the reference init, so the layers do something visible
        p.data.normal_(std=0.3)
    return m.eval()


@pytest.fixture
def model():
    return tiny()


TOK = bdh.ByteTokenizer()
PROMPT = "the cat sat"


def reference_forward(m, idx):
    """Pathway's ``BDH.forward``, transcribed, reading the weights from our model."""
    M, C = m.model, m.bdh_config
    B, T = idx.size()
    nh, D, N = C.n_head, C.n_embd, C.n_neurons
    ln = lambda t: F.layer_norm(t, (D,))
    x = ln(M.embed(idx).unsqueeze(1))
    for _ in range(C.n_layer):
        x_sparse = F.relu(x @ M.encoder)
        phases = torch.arange(T, dtype=torch.float32).view(1, 1, -1, 1) * M.freqs
        phases = (phases % 1) * (2 * math.pi)
        rot = lambda v: v * torch.cos(phases) + torch.stack((-v[..., 1::2], v[..., ::2]), -1).view(*v.size()) * torch.sin(phases)
        QR = rot(x_sparse)
        yKV = ln((QR @ QR.mT).tril(diagonal=-1) @ x)
        xy = x_sparse * F.relu(yKV @ M.encoder_v)
        y = ln(xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ M.decoder)
        x = ln(x + y)
    return x.view(B, T, D) @ m.lm_head.weight.T


def test_matches_reference_and_loads_reference_weights(model):
    idx = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(model(idx).logits, reference_forward(model, idx))
        ref_sd = {"encoder": model.model.encoder, "encoder_v": model.model.encoder_v, "decoder": model.model.decoder,
                  "attn.freqs": model.model.freqs, "embed.weight": model.model.embed.weight, "lm_head": model.lm_head.weight.T}
        loaded = bdh.from_reference_state_dict({"_orig_mod." + k: v for k, v in ref_sd.items()}, n_layer=3)
        torch.testing.assert_close(loaded(idx).logits, model(idx).logits)


def test_layers_share_weights(model):
    blocks = rsv.find_blocks(model)
    assert len(blocks) == 3 and blocks is model.model.layers
    assert all(b.shared is model.model for b in blocks)
    assert not any(k.startswith("model.layers") and "encoder" in k for k in model.state_dict())


def test_record_splits_the_update(model):
    rec = record(model, TOK, PROMPT)
    n = len(PROMPT)
    assert rec["h"].shape == (4, n, 16) and rec["tokens"][0] == "t"
    with torch.no_grad():
        logits = model(**TOK(PROMPT, return_tensors="pt")).logits[0]
    torch.testing.assert_close(rec["h"][-1] @ model.lm_head.weight.T, logits)
    # the first token has nothing to attend to, so its whole update is the layer norm's rescale (here ~0)
    assert rec["attn_write"][:, 0].abs().max() < 1e-5
    assert rec["attn_write"][:, 1:].norm(dim=-1).min() > 0
    torch.testing.assert_close(rec["attn_write"] + rec["mlp_write"], rec["h"][1:] - rec["h"][:-1])
    att = rec["attention"]
    assert att.shape == (3, 2, n, n)
    assert torch.triu(att).abs().max() == 0                     # strictly causal: no token attends to itself
    torch.testing.assert_close(att[:, :, 1:].sum(-1), torch.ones(3, 2, n - 1))
    stay, move = mixing(rec)
    torch.testing.assert_close(stay + move.sum(-1), torch.ones_like(stay))


def test_lens_and_trace(model):
    import json

    lens = rsv.fit_lens(model, TOK, ["the cat sat on a mat", "hello there world"], skip_first=2, progress=False)
    assert lens.layers == [1, 2, 3]
    h = record(model, TOK, PROMPT)["h"]
    with torch.no_grad():
        logits = model(**TOK(PROMPT, return_tensors="pt")).logits[0]
    torch.testing.assert_close(rsv.lens_logits(model, lens, h[3], 3), logits, rtol=1e-4, atol=1e-4)
    data = rsv.trace(model, TOK, PROMPT, lens)
    json.dumps(data)
    assert data["n_blocks"] == 3 and data["n_heads"] == 2 and data["parts"]["mlp"] == "+ norm rescale"
    assert data["contributions"] is False and data["special"] == [False] * len(PROMPT)


def test_edits_change_only_cells_above_and_right(model):
    base = record(model, TOK, PROMPT)
    vec = torch.randn(16) * 5
    for layer in (0, 2):
        rec = record(model, TOK, PROMPT, edits=[rsv.Edit(layer, 4, add=vec)])
        torch.testing.assert_close(rec["h"][layer, 4], base["h"][layer, 4] + vec)
        diff = (rec["h"] - base["h"]).norm(dim=-1)
        assert (diff[:layer] == 0).all() and (diff[:, :4] == 0).all() and diff[layer:, 4:].max() > 0
    d = rsv.word_direction(model, TOK, "z", 3)
    r = rsv.steer(model, TOK, PROMPT, d * 50, 3)
    assert r.prob("z")[1] > r.prob("z")[0]


def test_save_load_and_tokenizer(tmp_path, model):
    path = str(tmp_path / "m.pt")
    model.save(path)
    loaded, tok = bdh.load(path, device="cpu")
    ids = torch.tensor([list(b"hello")])
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits)
    assert loaded.config._name_or_path == path and rsv.load_model(path, device="cpu")[0].bdh_config == model.bdh_config
    assert TOK.decode(TOK("héllo")["input_ids"]) == "héllo"
    from residual_stream_vis.blocks import decode_tokens
    assert decode_tokens(TOK, TOK("é")["input_ids"]) == ["⟨c3⟩", "⟨a9⟩"]  # half a character each
    out = model.generate(ids, max_new_tokens=4)
    assert out.shape == (1, 9) and (out[:, :5] == ids).all()


def test_server_finds_the_lens_beside_the_checkpoint(tmp_path, model):
    import json, threading, urllib.request

    path = str(tmp_path / "m.pt")
    model.save(path)
    rsv.fit_lens(model, TOK, ["the cat sat on a mat"], skip_first=1, progress=False).save(bdh.lens_path_for(path))
    server = rsv.server.make_server(path, device="cpu", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(url).read().decode()
        assert '"lens": true' in page and "ROMEO" in page
        post = lambda p, b: json.loads(urllib.request.urlopen(urllib.request.Request(url + p, data=json.dumps(b).encode(), method="POST")).read())
        data = post("/api/trace", {"prompt": PROMPT})
        assert "jlens" in data["readouts"] and data["tokens"][3] == " "
        data = post("/api/generate", {"prompt": PROMPT, "max_new_tokens": 5, "chat": True})
        assert data["generated"]["start"] == len(PROMPT) and data["generated"]["n_new"] == 5 and not data["generated"]["chat"]
        d = post("/api/edit", {"prompt": PROMPT, "edit": {"layer": 2, "position": 5, "kind": "word", "word": "z", "amount": 2}})
        assert d["change"][2][5] > 0 and max(d["change"][2][:5]) == 0
        assert post("/api/track", {"prompt": PROMPT, "word": "cat"})["token"] == "c"
        assert '"neurons": true' in page
        nr = post("/api/neurons", {"prompt": PROMPT, "layer": 2, "position": 5, "edited": True})  # the edited run above
        assert nr["top"] and nr["top"][0]["acts"][2][5] > 0
    finally:
        server.shutdown()
        server.server_close()


def test_neuron_report(model):
    rec = record(model, TOK, PROMPT)
    r = model.neuron_report(TOK, PROMPT, 2, 5, k=4)
    assert r["n_neurons"] == 2 * 32 and 0 < r["writing"] <= r["active"] <= 1
    top = r["top"]
    assert 0 < len(top) <= 4 and [n["value"] for n in top] == sorted((n["value"] for n in top), reverse=True)
    assert all(len(n["writes"]) == 5 and len(n["excited_by"]) == 5 for n in top)
    n = top[0]
    assert n["acts"][0] is None and n["acts"][2][5] == pytest.approx(n["value"], abs=1e-3)
    assert all(row[0] == 0 for row in n["acts"][1:])  # the first token has no attention, so no neuron writes
    assert sum(x["share"] for x in top) <= 1 + 1e-3
    edited = model.neuron_report(TOK, PROMPT, 2, 5, edits=[rsv.Edit(1, 3, add=torch.randn(16) * 5)])
    assert edited["top"][0]["acts"] != n["acts"] or edited["top"][0]["id"] != n["id"]
    with pytest.raises(ValueError):
        model.neuron_report(TOK, PROMPT, 0, 5)
    assert rec["h"].shape[0] == 4
