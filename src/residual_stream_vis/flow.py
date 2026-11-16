"""Trace how information flows through a model for one prompt.

One forward pass records, for every block and token:

* the residual stream going in and coming out (``h``),
* what attention wrote into it and what the MLP wrote (``attention + mlp = h_out - h_in``),
* how much of the attention write came from each earlier token
  (attention weight × size of what that token passes on, as in
  :func:`residual_stream_vis.attention_contributions`).

From those, :func:`trace` builds the data the flow viewer draws, and
:func:`flow_weights` follows information back from one cell to the input (or
forward from one input token) through the residual stream and attention.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .blocks import _attention_projections, _first_tensor, decode_tokens, find_blocks, head_writes, token_str
from .edit import Edit, editing
from .lens import JacobianLens, _clean, _output_head, lens_logits


def _attention_output_module(block: nn.Module) -> nn.Module:
    """The module whose output is what attention adds to the residual stream.

    Gemma 2/3 normalize the attention output before adding it ("sandwich" norms,
    recognizable by ``pre_feedforward_layernorm``), so the write is the output of
    ``post_attention_layernorm``. In GPT-2, Llama and most others it's the
    attention module's own output (Llama's ``post_attention_layernorm`` is the
    norm *before* the MLP, despite the name).
    """
    if isinstance(getattr(block, "write", None), nn.Module):  # BDH: see residual_stream_vis.bdh
        return block.write
    if hasattr(block, "pre_feedforward_layernorm") and hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm
    for name in ("self_attn", "attn", "attention", "self_attention"):
        if hasattr(block, name):
            return getattr(block, name)
    raise ValueError(f"Can't find the attention module of a {type(block).__name__}")


@torch.no_grad()
def record(model: nn.Module, tokenizer, prompt: str, *, edits: Sequence[Edit] = (), layers_path: str | None = None) -> dict:
    """Run ``prompt`` once and record everything :func:`trace` needs. Special tokens are kept.

    With ``edits`` (see :mod:`residual_stream_vis.edit`), the run is edited and
    the recording shows the edited cells and everything recomputed from them.

    Returns a dict of CPU float tensors: ``h`` ``(n_blocks + 1, n, d)``,
    ``attn_write`` and ``mlp_write`` ``(n_blocks, n, d)``, ``attention``
    ``(n_blocks, heads, n, n)``, ``value_norms`` ``(n_blocks, heads, n)`` (or None
    for unsupported layouts), plus ``ids``, ``tokens`` and final ``logits``.
    """
    blocks = find_blocks(model, layers_path)
    device = next(model.parameters()).device
    hidden, attn_out, values = {}, {}, {}
    projections = [_attention_projections(b) for b in blocks]
    track_values = all(p is not None for p in projections)

    def block_hook(m, args, kwargs, out, i):
        if i == 0:
            hidden[0] = _first_tensor(args[0] if args else kwargs["hidden_states"])[0].float().cpu()
        hidden[i + 1] = _first_tensor(out)[0].float().cpu()

    handles = []
    was_training = model.training
    previous_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
    edit_ctx = editing(model, edits, layers_path=layers_path)
    edit_ctx.__enter__()  # edit hooks first, so the recording hooks see edited values
    try:
        if previous_attn != "eager":
            model.set_attn_implementation("eager")
        for i, b in enumerate(blocks):
            handles.append(b.register_forward_hook(lambda m, a, k, o, i=i: block_hook(m, a, k, o, i), with_kwargs=True))
            handles.append(_attention_output_module(b).register_forward_hook(
                lambda m, a, o, i=i: attn_out.__setitem__(i, _first_tensor(o)[0].float().cpu())))
            if track_values:
                mod, take, _ = projections[i]
                handles.append(mod.register_forward_hook(lambda m, a, o, i=i, take=take: values.__setitem__(i, take(o)[0])))
        model.eval()
        enc = tokenizer(prompt, return_tensors="pt")
        out = model(**{k: v.to(device) for k, v in enc.items()}, output_attentions=True)
    finally:
        model.train(was_training)
        if previous_attn not in (None, "eager"):
            model.set_attn_implementation(previous_attn)
        for h in handles:
            h.remove()
        edit_ctx.__exit__(None, None, None)

    n_blocks = len(blocks)
    h = torch.stack([hidden[i] for i in range(n_blocks + 1)])
    attn_write = torch.stack([attn_out[i] for i in range(n_blocks)])
    attention = torch.stack([a[0].float().cpu() for a in out.attentions])
    value_norms = None
    if track_values:
        value_norms = torch.stack([
            head_writes(values[i].float(), projections[i][2]().float(), attention.shape[1]).norm(dim=-1).cpu()
            for i in range(n_blocks)
        ])
    ids = enc["input_ids"][0].tolist()
    return dict(
        ids=ids, tokens=decode_tokens(tokenizer, ids), h=h, attn_write=attn_write,
        mlp_write=h[1:] - h[:-1] - attn_write, attention=attention, value_norms=value_norms,
        logits=out.logits[0].float().cpu(),
    )


def _shares(rec: dict) -> torch.Tensor:
    """``(blocks, heads, query, key)`` raw contributions: attention × what the key token passes on."""
    if rec["value_norms"] is None:
        return rec["attention"]
    return rec["attention"] * rec["value_norms"][:, :, None, :]


def mixing(rec: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """How each block mixes tokens: ``(stay, move)``.

    ``move[b, i, j]`` is the share of token ``i``'s state after block ``b`` that
    came from token ``j`` through attention, and ``stay[b, i]`` the share it kept
    from its own state. The split between keeping and moving uses the sizes of
    the residual (``|h_in|``) and of the attention write (``|attention|``); the
    split of the attention part between source tokens uses the contributions.
    The MLP works on each token separately, so it moves nothing between tokens.
    This is an approximation: vectors can partly cancel when added.
    """
    raw = _shares(rec).sum(dim=1)                                   # (blocks, q, k): combine heads
    share = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    resid = rec["h"][:-1].norm(dim=-1)
    attn = rec["attn_write"].norm(dim=-1)
    frac_attn = attn / (resid + attn).clamp_min(1e-12)              # (blocks, q)
    move = frac_attn[..., None] * share
    eye = torch.eye(move.shape[-1], dtype=torch.bool)
    stay = (1 - frac_attn) + move[:, eye]                           # attending to itself also stays
    move = move.masked_fill(eye, 0)
    return stay, move


def flow_weights(stay: torch.Tensor, move: torch.Tensor, layer: int, position: int, *, direction: str = "back") -> torch.Tensor:
    """How much each cell feeds into (``"back"``) or is fed by (``"forward"``) one cell.

    Rollout through the residual stream: going back one block, a cell's weight
    passes to its own earlier cell in proportion to ``stay`` and to the cells it
    attended to in proportion to ``move``.

    Args:
        stay, move: From :func:`mixing`.
        layer: Residual stream index of the starting cell (0 = embeddings).
        position: Token position of the starting cell.
        direction: ``"back"`` (where did this cell's information come from) or
            ``"forward"`` (where does this cell's information go).

    Returns:
        ``(n_blocks + 1, n_tokens)`` weights; the starting cell is 1.
    """
    n_blocks, n = stay.shape
    w = torch.zeros(n_blocks + 1, n)
    w[layer, position] = 1.0
    if direction == "back":
        for b in range(layer - 1, -1, -1):          # block b maps layer b -> b + 1
            nxt = w[b + 1]
            w[b] = nxt * stay[b] + nxt @ move[b]
    elif direction == "forward":
        for b in range(layer, n_blocks):
            cur = w[b]
            w[b + 1] = cur * stay[b] + move[b] @ cur
    else:
        raise ValueError(f"direction must be 'back' or 'forward', got {direction!r}")
    return w


def topk_rows(scores: torch.Tensor, k: int, block: int = 512):
    """Exact top-k along the last dim of a wide 2-D tensor; much faster than ``torch.topk`` on Apple GPUs.

    The k largest values lie in the k blocks with the largest maxima, so take the
    top k of the block maxima, then the top k within those blocks (and the tail
    that doesn't fill a block). Ties may be broken differently from ``torch.topk``.
    """
    n, V = scores.shape
    nb = V // block
    if nb <= k:
        return scores.topk(k, -1)
    blocks = scores.as_strided((n, nb, block), (scores.stride(0), block, 1))
    bi = blocks.amax(-1).topk(k, -1).indices                                  # (n, k)
    cand = blocks.gather(1, bi[..., None].expand(-1, -1, block)).reshape(n, k * block)
    cidx = (bi[..., None] * block + torch.arange(block, device=scores.device)).reshape(n, k * block)
    if V > nb * block:
        cand = torch.cat([cand, scores[:, nb * block:]], 1)
        cidx = torch.cat([cidx, torch.arange(nb * block, V, device=scores.device).expand(n, -1)], 1)
    v, j = cand.topk(k, -1)
    return torch.return_types.topk((v, cidx.gather(1, j)))


@torch.no_grad()
def read_cells(model: nn.Module, lens: JacobianLens | None, h: torch.Tensor, layers: Sequence[int], *,
               k: int = 5, track_ids: Sequence[int] = (), chunk: int = 1024):
    """Read every cell of ``layers`` with one lens in a few large GPU batches.

    Returns ``(probs, ids, ranks)``: the top-``k`` probabilities and token ids,
    ``(len(layers), n, k)`` (``None`` when ``k == 0``), and the rank of each
    token in ``track_ids`` (1 = top), ``(len(layers), n, len(track_ids))``
    (``None`` without ``track_ids``). Same numbers as :func:`lens_logits`
    followed by softmax and top-k, computed in one pass over the vocabulary.
    """
    norm, w_u, softcap = _output_head(model)
    n = h.shape[1]
    xs = []
    for l in layers:
        J = lens.matrix(l) if lens is not None else None
        xs.append(h[l].float() if J is None else h[l].float() @ J.T)
    x = norm(torch.cat(xs).to(w_u.device, w_u.dtype))
    ids = torch.tensor(list(track_ids), device=w_u.device, dtype=torch.long)
    probs, top_ids, ranks = [], [], []
    for s in range(0, x.shape[0], chunk):
        lg = x[s:s + chunk] @ w_u.T
        if softcap:
            lg = softcap * torch.tanh(lg / softcap)
        if k:
            top = topk_rows(lg, k)
            lse = torch.logsumexp(lg.float(), dim=-1, keepdim=True)
            probs.append((top.values.float() - lse).exp())
            top_ids.append(top.indices)
        if len(ids):
            ref = lg[:, ids]
            ranks.append(torch.stack([(lg > ref[:, j:j + 1]).sum(-1) + 1 for j in range(len(ids))], -1))
    shape = (len(layers), n)
    probs = torch.cat(probs).cpu().view(*shape, k) if k else None
    top_ids = torch.cat(top_ids).cpu().view(*shape, k) if k else None
    ranks = torch.cat(ranks).cpu().view(*shape, len(ids)) if len(ids) else None
    return probs, top_ids, ranks


def _word_info(tokenizer, word: str) -> tuple[int, str, int]:
    """``(token id, token text, how many tokens the word is)`` for highlighting ``word``."""
    from .edit import word_token

    i, text = word_token(tokenizer, word)
    pieces = [t for t in tokenizer(word.strip(), add_special_tokens=False)["input_ids"] if token_str(tokenizer, t).strip()]
    n_tokens = 1 if len(tokenizer(word, add_special_tokens=False)["input_ids"]) == 1 else max(1, len(pieces))
    return i, text, n_tokens


_LABELS: dict[tuple[int, int], str] = {}


def _label(tokenizer, i: int) -> str:
    """Cached display text for one token id (decoding is the slow part of long readouts)."""
    key = (id(tokenizer), i)
    if key not in _LABELS:
        _LABELS[key] = _clean(token_str(tokenizer, i))
    return _LABELS[key]


def _top(logits: torch.Tensor, tokenizer, k: int):
    probs = logits.softmax(-1)
    top = probs.topk(k, dim=-1)
    top = type(top)((top.values.cpu(), top.indices.cpu()))
    return [[[_label(tokenizer, i) for i in top.indices[p].tolist()],
             [round(v, 4) for v in top.values[p].tolist()]] for p in range(len(logits))]


@torch.no_grad()
def trace(
    model: nn.Module,
    tokenizer,
    prompt: str,
    lens: JacobianLens | None = None,
    *,
    top_k: int = 5,
    edits: Sequence[Edit] = (),
    layers_path: str | None = None,
) -> dict:
    """Everything the flow viewer shows for one prompt, as plain JSON-ready data.

    Args:
        model, tokenizer: A causal LM and its tokenizer.
        prompt: The text to run.
        lens: A :class:`~residual_stream_vis.JacobianLens` for the J-lens readout,
            or ``None`` for the logit lens only.
        top_k: Top words kept per cell for each lens.
        edits: Edits to apply during the run (positions index the input tokens,
            special tokens included); see :mod:`residual_stream_vis.edit`.
        layers_path: Dotted path to the blocks if auto-detection picks the wrong list.

    Returns:
        A dict with ``tokens``, ``n_blocks``, ``n_heads``, per-cell lens readouts
        (``logit`` and, with a lens, ``jlens``; ``[layer][position] = [words, probs]``),
        per-block sizes (``resid``, ``attn``, ``mlp``: ``[block][position]``),
        token mixing (``stay[block][position]``, and ``move[block][query]`` as
        ``[key, share]`` pairs above ``MOVE_MIN``), per-head contributions
        (``heads[block][head][query][key]``; for each query, the values over all
        heads and keys sum to 1, and summing over heads gives the combined share
        of the attention write from each key; ``None`` past ``HEADS_INLINE_MAX``
        tokens, see :func:`heads_at`),
        and the model's ``output`` at the last position.
    """
    rec = record(model, tokenizer, prompt, edits=edits, layers_path=layers_path)
    return trace_record(rec, model, tokenizer, prompt, lens, top_k=top_k)


#: How the viewer names the two parts of a block's update (models can override with ``model.update_parts``).
UPDATE_PARTS = dict(attn="+ attention", mlp="+ MLP",
                    note="Vector sizes. Only attention moves information between tokens; the MLP works on this token alone.")

#: Prompts longer than this leave per-head shares out of :func:`trace` (they grow
#: with ``layers × heads × tokens²``); the viewer fetches them one cell at a time.
HEADS_INLINE_MAX = 64
#: Token-mixing shares below this are dropped from the sparse ``move`` lists.
MOVE_MIN = 1e-3


def heads_at(rec: dict, block: int, position: int) -> list[list[float]]:
    """Per-head shares of one cell's attention write, ``[head][key]``: summing over heads gives the combined share."""
    raw = _shares(rec)[block, :, position]                         # (heads, keys)
    raw = raw / raw.sum().clamp_min(1e-12)
    return [[round(float(v), 4) for v in row] for row in raw]


@torch.no_grad()
def track(rec: dict, model: nn.Module, tokenizer, word: str, lens: JacobianLens | None = None) -> dict:
    """Rank of ``word``'s token in every cell, under each lens (1 = the lens's top word).

    Returns ``{"token": ..., "n_tokens": ..., "logit": [[rank per position] per layer], "jlens": ...}``;
    J-lens layers it doesn't cover are ``None``. A word that is several tokens is
    tracked by its first token (``n_tokens`` says how many it has).
    """
    i, text, n_tokens = _word_info(tokenizer, word)
    n_layers = rec["h"].shape[0]

    def ranks(L, layers):
        return read_cells(model, L, rec["h"], layers, k=0, track_ids=[i])[2][..., 0].tolist()

    out = {"token": text, "n_tokens": n_tokens, "logit": ranks(None, list(range(n_layers)))}
    if lens is not None:
        by_layer = dict(zip(lens.layers, ranks(lens, lens.layers)))
        out["jlens"] = [by_layer.get(l) for l in range(n_layers)]
    return out


@torch.no_grad()
def trace_record(rec: dict, model: nn.Module, tokenizer, prompt: str, lens: JacobianLens | None = None, *, top_k: int = 5,
                 track_word: str | None = None) -> dict:
    """:func:`trace` from an existing :func:`record` result.

    With ``track_word``, the result also has ``track`` (as from :func:`track`),
    computed in the same pass as the readouts.
    """
    n_blocks, n_heads = rec["attention"].shape[:2]
    r3 = lambda t: [[round(float(v), 4) for v in row] for row in t]

    tracked, tids = None, []
    if track_word and track_word.strip():
        tid, ttext, n_pieces = _word_info(tokenizer, track_word)
        tracked, tids = {"token": ttext, "n_tokens": n_pieces}, [tid]

    def cells(L, layers):
        probs, ids, ranks = read_cells(model, L, rec["h"], layers, k=top_k, track_ids=tids)
        rows = [[[[_label(tokenizer, int(i)) for i in ids[a, p]], [round(float(v), 4) for v in probs[a, p]]]
                 for p in range(ids.shape[1])] for a in range(len(layers))]
        return rows, (ranks[..., 0].tolist() if ranks is not None else None)

    all_layers = list(range(n_blocks + 1))
    readouts = {}
    readouts["logit"], logit_ranks = cells(None, all_layers)
    if tracked:
        tracked["logit"] = logit_ranks
    if lens is not None:
        j_rows, j_ranks = cells(lens, lens.layers)
        by_layer = dict(zip(lens.layers, j_rows))
        readouts["jlens"] = [by_layer.get(l) for l in all_layers]
        if tracked:
            rk = dict(zip(lens.layers, j_ranks))
            tracked["jlens"] = [rk.get(l) for l in all_layers]
    stay, move = mixing(rec)
    n = rec["h"].shape[1]
    heads = None
    if n <= HEADS_INLINE_MAX:
        heads = _shares(rec)  # normalized per query over heads *and* keys, so summing heads gives the combined share
        heads = heads / heads.sum(dim=(1, 3), keepdim=True).clamp_min(1e-12)
        heads = [[r3(hq) for hq in hb] for hb in heads]
    # sparse token mixing: for each block and query, [key, share] pairs above MOVE_MIN
    move_sparse = [[[[j, round(float(v), 4)] for j, v in enumerate(row[:i]) if v >= MOVE_MIN] for i, row in enumerate(mb.tolist())]
                   for mb in move]
    final = _top(rec["logits"][-1:], tokenizer, top_k)[0]
    return dict(
        prompt=prompt,
        model=getattr(getattr(model, "config", None), "_name_or_path", None),
        tokens=[_clean(t) for t in rec["tokens"]],
        special=[i in set(tokenizer.all_special_ids) for i in rec["ids"]],
        n_blocks=int(n_blocks), n_heads=int(n_heads),
        readouts=readouts,
        resid=r3(rec["h"][:-1].norm(dim=-1)), attn=r3(rec["attn_write"].norm(dim=-1)), mlp=r3(rec["mlp_write"].norm(dim=-1)),
        stay=r3(stay), move=move_sparse, heads=heads,
        output=final,
        contributions=rec["value_norms"] is not None,
        parts=getattr(model, "update_parts", UPDATE_PARTS),
        track=tracked,
    )
