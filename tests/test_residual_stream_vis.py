import pytest
import torch
import transformers

import residual_stream_vis as rsv


class CharTokenizer:
    """Offline stand-in for a HF tokenizer: one token per character, plus a BOS."""

    bos_token_id = 1
    all_special_ids = [0, 1]

    def __call__(self, text, return_tensors=None, add_special_tokens=True, truncation=False, max_length=None):
        ids = ([self.bos_token_id] if add_special_tokens else []) + [2 + (ord(c) - 32) % 98 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        if return_tensors is None:
            return {"input_ids": ids}
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids):
        return "".join("<bos>" if i == 1 else chr(i - 2 + 32) for i in ids)


def tiny_models():
    common = dict(vocab_size=100, num_hidden_layers=3, num_attention_heads=2, bos_token_id=1, eos_token_id=0, pad_token_id=0)
    return {
        "gpt2": transformers.GPT2LMHeadModel(
            transformers.GPT2Config(vocab_size=100, bos_token_id=1, eos_token_id=0, n_layer=3, n_head=2, n_embd=16, n_positions=64)
        ),
        "llama": transformers.LlamaForCausalLM(
            transformers.LlamaConfig(hidden_size=16, intermediate_size=32, num_key_value_heads=2, **common)
        ),
        "gemma3": transformers.Gemma3ForCausalLM(
            transformers.Gemma3TextConfig(
                hidden_size=16, intermediate_size=32, num_key_value_heads=1, head_dim=8, **common
            )
        ),
    }


MODELS = tiny_models()
PROMPTS = ["hello there", "the cat sat"]


@pytest.fixture(params=list(MODELS))
def model(request):
    torch.manual_seed(0)
    return MODELS[request.param].eval()


def test_find_blocks(model):
    assert len(rsv.find_blocks(model)) == 3


def test_record_matches_hidden_states(model):
    from residual_stream_vis.flow import record

    tok = CharTokenizer()
    rec = record(model, tok, PROMPTS[0])
    assert rec["h"].shape[:2] == (4, len(PROMPTS[0]) + 1) and rec["tokens"][0] == "<bos>"
    with torch.no_grad():
        hidden = model(**tok(PROMPTS[0], return_tensors="pt"), output_hidden_states=True).hidden_states
    # HF applies the final norm to its last hidden state; the hooks record the raw
    # residual stream, so only the layers before that must match exactly.
    for i in range(len(hidden) - 1):
        torch.testing.assert_close(rec["h"][i], hidden[i][0].float(), rtol=1e-4, atol=1e-5)


def test_record_cleans_up_and_records_attention(model):
    from residual_stream_vis.flow import record

    # transformers may install its own hooks, so compare against what was there before.
    before = [dict(b._forward_hooks) for b in rsv.find_blocks(model)]
    impl = model.config._attn_implementation
    rec = record(model, CharTokenizer(), PROMPTS[0])
    assert [dict(b._forward_hooks) for b in rsv.find_blocks(model)] == before
    assert model.config._attn_implementation == impl
    att, n = rec["attention"], len(PROMPTS[0]) + 1
    assert att.shape == (3, 2, n, n)
    torch.testing.assert_close(att.sum(-1), torch.ones(3, 2, n))  # rows are distributions
    assert torch.triu(att, diagonal=1).abs().max() == 0          # causal


def test_pca_matches_numpy_variance():
    from residual_stream_vis.flow import pca

    torch.manual_seed(0)
    x = torch.randn(50, 10) @ torch.randn(10, 10)
    proj, ratio = pca(x, 3)
    centered = (x - x.mean(0)).double()
    eig = torch.linalg.eigvalsh(centered.T @ centered).flip(0)
    torch.testing.assert_close(ratio.double(), (eig[:3] / eig.sum()), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(proj.var(0).double(), eig[:3] / 49, rtol=1e-4, atol=1e-6)
    proj, ratio = pca(torch.randn(2, 8), 3)  # fewer samples than components
    assert proj.shape == (2, 3) and ratio.shape == (3,)


def test_head_writes_reconstruct_attention_output(model):
    """sum_j attention[h, i, j] * writes[h, j], summed over heads, must equal the real attention output."""
    from residual_stream_vis.blocks import _attention_projections, head_writes
    from residual_stream_vis.flow import record

    tok = CharTokenizer()
    block = rsv.find_blocks(model)[1]
    v_mod, take, out_w = _attention_projections(block)
    out_proj = next(m for n, m in block.named_modules() if n.endswith(("o_proj", "attn.c_proj")))
    seen = {}
    hs = [v_mod.register_forward_hook(lambda m, a, o: seen.__setitem__("v", take(o)[0])),
          out_proj.register_forward_hook(lambda m, a, o: seen.__setitem__("out", o[0]))]
    try:
        rec = record(model, tok, PROMPTS[0])
    finally:
        for h in hs:
            h.remove()
    att = rec["attention"][1]                                     # (heads, q, k)
    writes = head_writes(seen["v"].float(), out_w().float(), att.shape[0])
    rebuilt = torch.einsum("hqk,hkd->qd", att, writes)
    if out_proj.bias is not None:
        rebuilt = rebuilt + out_proj.bias
    torch.testing.assert_close(rebuilt, seen["out"].float(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rec["value_norms"][1], writes.norm(dim=-1))


# ------------------------------------------------------------------ J-lens


def brute_force_jacobian(model, tok, prompt, source_block, skip_first):
    """The fit_lens estimator computed the slow, obvious way with full autograd Jacobians."""
    blocks = rsv.find_blocks(model)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        base = {}
        hs = [b.register_forward_hook(lambda m, a, o, i=i: base.__setitem__(i, o if torch.is_tensor(o) else o[0]))
              for i, b in enumerate(blocks)]
        model(input_ids=ids)
        for h in hs:
            h.remove()

    def run(h_src):
        def replace(m, a, o):
            return h_src[None] if torch.is_tensor(o) else (h_src[None],) + tuple(o[1:])
        out = {}
        h1 = blocks[source_block].register_forward_hook(replace)
        h2 = blocks[-1].register_forward_hook(lambda m, a, o: out.__setitem__("f", o if torch.is_tensor(o) else o[0]))
        try:
            model(input_ids=ids)
        finally:
            h1.remove(); h2.remove()
        return out["f"][0]

    jac = torch.autograd.functional.jacobian(run, base[source_block][0].clone())  # (seq, d, seq, d)
    valid = torch.arange(skip_first, ids.shape[1] - 1)
    summed = jac[valid].sum(0)              # (d_out, seq_src, d_in): sum over target positions
    return summed[:, valid].mean(1)          # mean over source positions


def test_fit_lens_matches_brute_force(model):
    tok = CharTokenizer()
    prompts = ["the cat sat on a mat", "hello there world"]
    lens = rsv.fit_lens(model, tok, prompts, dim_batch=3, skip_first=2, progress=False)
    assert sorted(lens.jacobians) == [0, 1] and lens.n_prompts == 2
    for block in (0, 1):
        expected = sum(brute_force_jacobian(model, tok, p, block, 2) for p in prompts) / 2
        torch.testing.assert_close(lens.jacobians[block], expected, rtol=1e-4, atol=1e-5)
    assert all(p.requires_grad for p in model.parameters())  # restored


def test_lens_final_layer_is_model_output(model):
    tok = CharTokenizer()
    lens = rsv.fit_lens(model, tok, ["the cat sat on a mat"], skip_first=1, progress=False)
    from residual_stream_vis.flow import record

    h = record(model, tok, PROMPTS[0])["h"]
    with torch.no_grad():
        logits = model(**tok(PROMPTS[0], return_tensors="pt")).logits[0].float()
    torch.testing.assert_close(rsv.lens_logits(model, lens, h[3], 3), logits, rtol=1e-4, atol=1e-4)
    assert lens.layers == [1, 2, 3] and lens.matrix(3) is None


def test_lens_save_load(tmp_path, model):
    lens = rsv.fit_lens(model, CharTokenizer(), ["the cat sat on a mat"], skip_first=1, progress=False)
    lens.save(tmp_path / "l.pt")
    loaded = rsv.JacobianLens.load(tmp_path / "l.pt")
    torch.testing.assert_close(loaded.jacobians[1], lens.jacobians[1], rtol=1e-3, atol=1e-3)  # fp16 on disk


def _mlp_out(name, block):
    return block.post_feedforward_layernorm if name == "gemma3" else block.mlp


@pytest.mark.parametrize("name", list(MODELS))
def test_flow_record_matches_hooks(name):
    from residual_stream_vis.flow import mixing, record

    model, tok = MODELS[name].eval(), CharTokenizer()
    rec = record(model, tok, "hello there")
    got = {}
    blocks = rsv.find_blocks(model)
    hs = [_mlp_out(name, b).register_forward_hook(lambda m, a, o, i=i: got.__setitem__(i, (o if torch.is_tensor(o) else o[0])[0].float()))
          for i, b in enumerate(blocks)]
    with torch.no_grad():
        model(**tok("hello there", return_tensors="pt"))
    for h in hs:
        h.remove()
    for i in got:  # attention + MLP writes add up to the block's update, and the MLP part is the MLP's output
        torch.testing.assert_close(rec["mlp_write"][i], got[i], rtol=1e-4, atol=1e-5)
    stay, move = mixing(rec)
    torch.testing.assert_close(stay + move.sum(-1), torch.ones_like(stay))
    assert (move.triu() == 0).all()  # nothing flows from later tokens


def test_flow_weights_conserve_and_trace_json(model):
    import json
    from residual_stream_vis.flow import mixing, record

    tok = CharTokenizer()
    rec = record(model, tok, "the cat sat")
    stay, move = mixing(rec)
    n_blocks, n = stay.shape
    back = rsv.flow_weights(stay, move, n_blocks, n - 1)
    torch.testing.assert_close(back.sum(dim=1), torch.ones(n_blocks + 1))
    fwd = rsv.flow_weights(stay, move, 0, 1, direction="forward")
    assert (fwd[:, 0] == 0).all() and fwd[0, 1] == 1  # the first token can't receive from token 1
    lens = rsv.fit_lens(model, tok, ["the cat sat on a mat"], skip_first=1, progress=False)
    data = rsv.trace(model, tok, "the cat sat", lens)
    json.dumps(data)
    assert data["readouts"]["jlens"][0] is None and len(data["readouts"]["jlens"][n_blocks]) == n
    for b in range(n_blocks):  # summing heads gives each query's combined share, which sums to 1
        row = [sum(h[n - 1][j] for h in data["heads"][b]) for j in range(n)]
        assert sum(row) == pytest.approx(1, abs=1e-3)


def test_flow_server_and_static_page(tmp_path):
    import json, threading, urllib.request

    model, tok = MODELS["gpt2"].eval(), CharTokenizer()
    server = rsv.server.make_server(model, tok, lens=None, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(url).read().decode()
        assert "Residual Stream Flow" in page and '"live": true' in page
        req = urllib.request.Request(url + "/api/trace", data=json.dumps({"prompt": "hi there"}).encode(), method="POST")
        data = json.loads(urllib.request.urlopen(req).read())
        assert data["tokens"][0] == "<bos>" and data["n_blocks"] == 3
        assert len(data["pca"]["all"]) == 4 and data["pca"]["skip_first"][2]["xyz"][0] is None
        assert all(len(c) == 3 for c in data["pca"]["all"][2]["xyz"])
        assert len(urllib.request.urlopen(url + "/plotly.min.js").read()) > 1_000_000
    finally:
        server.shutdown()
        server.server_close()
    html = rsv.flow_html(data, tmp_path / "flow.html")
    assert (tmp_path / "flow.html").exists() and '"live": false' in html and "cdn.jsdelivr.net" in html
    assert "<!--RSV_PLOTLY-->" not in rsv.flow_html(data, plotly="inline")


def test_patch_identity_and_last_layer_transplant(model):
    tok = CharTokenizer()
    same = rsv.patch(model, tok, "the cat sat", 2, source="the cat sat")  # copy a cell onto itself
    torch.testing.assert_close(same.after, same.before)
    n = rsv.find_blocks(model).__len__()
    # replacing the last-layer cell at the last position makes the output the donor's output
    r = rsv.patch(model, tok, "the cat sat", n, source="hello there")
    donor = rsv.patch(model, tok, "hello there", 2, source="hello there")
    torch.testing.assert_close(r.after, donor.before, rtol=1e-4, atol=1e-5)
    assert len(r.top(3)) == 3 and "before" in repr(r)


def test_edit_changes_only_cells_above_and_right(model):
    from residual_stream_vis.flow import record

    tok = CharTokenizer()
    prompt = "the cat sat"
    base = record(model, tok, prompt)
    vec = torch.randn(base["h"].shape[-1]) * 5
    for layer in (0, 2):
        pos = 4
        rec = record(model, tok, prompt, edits=[rsv.Edit(layer, pos, add=vec)])
        torch.testing.assert_close(rec["h"][layer, pos], base["h"][layer, pos] + vec)
        diff = (rec["h"] - base["h"]).norm(dim=-1)
        assert (diff[:layer] == 0).all()               # layers below: untouched
        assert (diff[:, :pos] == 0).all()              # earlier tokens: untouched
        assert diff[layer:, pos:].max() > 0
    # steering by zero, and the steering vector between a prompt and itself, change nothing
    z = rsv.steer(model, tok, prompt, torch.zeros(base["h"].shape[-1]), 1)
    torch.testing.assert_close(z.after, z.before)
    sv = rsv.steering_vector(model, tok, [prompt, "hello"], [prompt, "hello"], 1)
    assert sv.abs().max() == 0
    # positions skip special tokens by default: position 0 is the first token after <bos>
    torch.testing.assert_close(rsv.cell(model, tok, prompt, 2, 0), base["h"][2, 1])
    torch.testing.assert_close(rsv.cell(model, tok, prompt, 2, 0, skip_special_tokens=False), base["h"][2, 0])


def test_word_direction_pushes_toward_the_word(model):
    tok = CharTokenizer()
    n = len(rsv.find_blocks(model))
    d = rsv.word_direction(model, tok, "z", n)
    assert d.norm() == pytest.approx(1, abs=1e-5)
    h = rsv.cell(model, tok, "the cat sat", n)
    r = rsv.steer(model, tok, "the cat sat", d, n, alpha=50 * float(h.norm()))
    before, after = r.prob("z")
    assert after > before


def test_flow_server_edit_endpoint():
    import json, threading, urllib.request

    model, tok = MODELS["llama"].eval(), CharTokenizer()
    server = rsv.server.make_server(model, tok, lens=None, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/api/edit"

    def post(edit):
        req = urllib.request.Request(url, data=json.dumps({"prompt": "the cat sat", "edit": edit}).encode(), method="POST")
        try:
            return 200, json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        code, d = post({"layer": 2, "position": 5, "kind": "word", "word": "z", "amount": 2})
        assert code == 200 and d["edit"]["layer"] == 2 and "toward" in d["edit"]["description"]
        assert max(d["change"][1]) == 0 and d["change"][2][5] > 0 and max(d["change"][2][:5]) == 0
        code, d = post({"layer": 1, "position": 3, "kind": "patch", "source": "hello"})
        assert code == 200 and d["change"][1][3] > 0
        code, d = post({"layer": 1, "position": 3, "kind": "diff", "prompt_a": "dog", "prompt_b": "cat", "amount": 1})
        assert code == 200
        code, d = post({"layer": 2, "position": 5, "kind": "swap", "old_word": "", "new_word": "z", "direction": "unembed"})
        assert code == 200 and "Swapped" in d["edit"]["description"] and d["change"][2][5] > 0
        code, d = post({"layer": 2, "position": 5, "kind": "word", "word": "z", "direction": "jlens"})
        assert code == 400 and "J-lens" in d["error"]  # this server has no lens
        code, d = post({"layer": 1, "position": 3, "kind": "patch", "source": ""})
        assert code == 400 and "donor" in d["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_layer_pca(model):
    from residual_stream_vis.flow import layer_pca, pca, record

    h = record(model, CharTokenizer(), "the cat sat")["h"]
    got = layer_pca(h)
    want, ratio = pca(torch.nn.functional.normalize(h[2], dim=-1), 3)
    torch.testing.assert_close(torch.tensor(got[2]["xyz"]), want, atol=1e-3, rtol=1e-3)
    assert got[2]["var"] == pytest.approx(ratio.tolist(), abs=1e-3)
    skip = layer_pca(h, skip_first=True)
    assert skip[2]["xyz"][0] is None and len(skip[2]["xyz"]) == h.shape[1]


def test_swap_and_direction_kinds(model):
    tok = CharTokenizer()
    lens = rsv.fit_lens(model, tok, ["the cat sat on a mat"], skip_first=1, progress=False)
    dj = rsv.word_direction(model, tok, "z", 2, lens)                       # J-lens by default with a lens
    du = rsv.word_direction(model, tok, "z", 2, lens, kind="unembed")
    assert dj.norm() == pytest.approx(1, abs=1e-5) and not torch.allclose(dj, du)
    with pytest.raises(ValueError, match="needs a lens"):
        rsv.word_direction(model, tok, "z", 2, kind="jlens")
    with pytest.raises(ValueError, match="covers layers"):
        rsv.word_direction(model, tok, "z", 0, lens)
    h = torch.randn(16)
    old, new = rsv.word_direction(model, tok, "a", 2, lens), rsv.word_direction(model, tok, "b", 2, lens)
    v = rsv.edit.swap_vector(h, old, new)
    assert float(v @ old) == pytest.approx(float(2 * (h @ old).abs() * (new @ old)), abs=1e-4)  # old component removed
    r = rsv.swap(model, tok, "a cat", "a", "b", 2, 0, lens=lens)
    assert r.after.shape == r.before.shape and not torch.allclose(r.after, r.before)


def test_partial_character_tokens_are_labeled_by_bytes():
    from residual_stream_vis.blocks import _gpt2_byte_chars, token_str

    table = _gpt2_byte_chars()
    assert len(table) == 256 and sorted(table.values()) == list(range(256))
    inv = {b: c for c, b in table.items()}

    class ByteTok:  # GPT-2-style: each token is a string of byte characters
        pieces = {0: inv[0x20] + inv[0xE6], 1: inv[0x97] + inv[0xA5], 2: "ok"}
        def decode(self, ids):
            return bytes(table[c] for i in ids for c in self.pieces[i]).decode("utf-8", errors="replace")
        def convert_ids_to_tokens(self, i):
            return self.pieces[i]

    tok = ByteTok()
    assert [token_str(tok, i) for i in range(3)] == [" ⟨e6⟩", "⟨97 a5⟩", "ok"]


def test_examples_fit_the_tokenizer():
    from residual_stream_vis.server import _fit_examples

    class Tok:
        def __init__(self, split): self.split = split
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [1, 2] if self.split and text.startswith(" ") else [1]}

    ex = ["Q: legs?\nA: ", "The city of"]
    assert _fit_examples(Tok(split=True), ex) == ex                       # Gemma-like: keep the space
    assert _fit_examples(Tok(split=False), ex) == ["Q: legs?\nA:", "The city of"]  # GPT-2-like: drop it


def test_cli_accepts_old_serve_flag(monkeypatch):
    import residual_stream_vis.server as server_mod
    from residual_stream_vis.__main__ import main

    calls = []
    monkeypatch.setattr(server_mod, "serve", lambda *a, **k: calls.append((a, k)))
    main(["gpt2", "--serve", "--no-browser", "--lens", "none", "Hello there"])
    (args, kw), = calls
    assert args == ("gpt2",) and kw["lens"] is None and kw["examples"] == ["Hello there"] and not kw["open_browser"]
