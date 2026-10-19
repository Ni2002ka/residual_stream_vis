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
import torch.nn.functional as F
from torch import nn

from .blocks import _attention_projections, _first_tensor, decode_tokens, find_blocks, head_writes, token_str
from .edit import Edit, editing
from .lens import JacobianLens, _clean, lens_logits


def _attention_output_module(block: nn.Module) -> nn.Module:
    """The module whose output is what attention adds to the residual stream.

    Gemma 2/3 normalize the attention output before adding it ("sandwich" norms,
    recognizable by ``pre_feedforward_layernorm``), so the write is the output of
    ``post_attention_layernorm``. In GPT-2, Llama and most others it's the
    attention module's own output (Llama's ``post_attention_layernorm`` is the
    norm *before* the MLP, despite the name).
    """
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


def _top(logits: torch.Tensor, tokenizer, k: int):
    probs = logits.softmax(-1)
    top = probs.topk(k, dim=-1)
    return [[[_clean(token_str(tokenizer, i)) for i in top.indices[p].tolist()],
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
        token mixing (``stay[block][position]``, ``move[block][query][key]``),
        per-head contributions (``heads[block][head][query][key]``; for each
        query, the values over all heads and keys sum to 1, and summing over
        heads gives the combined share of the attention write from each key)
        the model's ``output`` at the last position, and a per-layer ``pca`` of
        the token vectors (``all`` tokens, and ``skip_first`` without the first;
        see :func:`layer_pca`).
    """
    rec = record(model, tokenizer, prompt, edits=edits, layers_path=layers_path)
    return trace_record(rec, model, tokenizer, prompt, lens, top_k=top_k)


def pca(x: torch.Tensor, n_components: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA via SVD in torch.

    Args:
        x: ``(n_samples, n_features)`` data.
        n_components: Number of principal components to keep.

    Returns:
        ``(projected, explained_variance_ratio)`` where ``projected`` is
        ``(n_samples, n_components)``. Component signs are fixed so the
        largest-magnitude loading of each component is positive, which keeps
        the output deterministic.
    """
    x = x.double()
    centered = x - x.mean(dim=0, keepdim=True)
    # Economy SVD: with n_tokens << d_model this is cheap.
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    k = min(n_components, s.numel())
    vh = vh[:k]
    signs = torch.sign(vh.gather(1, vh.abs().argmax(dim=1, keepdim=True))).squeeze(1)
    signs[signs == 0] = 1
    projected = (u[:, :k] * s[:k]) * signs
    var = s**2
    ratio = var[:k] / var.sum().clamp_min(1e-30)
    if k < n_components:  # fewer samples than components: pad with zeros
        projected = F.pad(projected, (0, n_components - k))
        ratio = F.pad(ratio, (0, n_components - k))
    return projected.float(), ratio.float()


def layer_pca(h: torch.Tensor, *, skip_first: bool = False) -> list[dict]:
    """Per-layer PCA of one prompt's token vectors, for the viewer's PCA panel.

    Each layer gets its own PCA on the L2-normalized vectors (so it shows
    direction, not size). Axes are
    refit at every layer, so compare tokens within a layer, not across layers.

    Args:
        h: ``(n_layers, n_tokens, d_model)`` residual stream.
        skip_first: Leave out the first token (often an outlier "attention sink")
            when fitting; its coordinates are then ``None``.

    Returns:
        One dict per layer: ``xyz`` (``[pc1, pc2, pc3]`` per token, or ``None``)
        and ``var`` (explained variance ratio of each component).
    """
    out = []
    start = 1 if skip_first and h.shape[1] > 2 else 0
    for layer in range(h.shape[0]):
        x = torch.nn.functional.normalize(h[layer, start:], dim=-1)
        pcs, ratio = pca(x, 3)
        xyz = [None] * start + [[round(float(v), 4) for v in row] for row in pcs]
        out.append(dict(xyz=xyz, var=[round(float(v), 4) for v in ratio]))
    return out


@torch.no_grad()
def trace_record(rec: dict, model: nn.Module, tokenizer, prompt: str, lens: JacobianLens | None = None, *, top_k: int = 5) -> dict:
    """:func:`trace` from an existing :func:`record` result."""
    n_blocks, n_heads = rec["attention"].shape[:2]
    r3 = lambda t: [[round(float(v), 4) for v in row] for row in t]

    readouts = {"logit": [_top(lens_logits(model, None, rec["h"][l], l), tokenizer, top_k) for l in range(n_blocks + 1)]}
    if lens is not None:
        readouts["jlens"] = [
            None if l not in lens.layers else _top(lens_logits(model, lens, rec["h"][l], l), tokenizer, top_k)
            for l in range(n_blocks + 1)
        ]
    stay, move = mixing(rec)
    heads = _shares(rec)  # normalized per query over heads *and* keys, so summing heads gives the combined share
    heads = heads / heads.sum(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    final = _top(rec["logits"][-1:], tokenizer, top_k)[0]
    return dict(
        prompt=prompt,
        model=getattr(getattr(model, "config", None), "_name_or_path", None),
        tokens=[_clean(t) for t in rec["tokens"]],
        special=[i in set(tokenizer.all_special_ids) for i in rec["ids"]],
        n_blocks=int(n_blocks), n_heads=int(n_heads),
        readouts=readouts,
        resid=r3(rec["h"][:-1].norm(dim=-1)), attn=r3(rec["attn_write"].norm(dim=-1)), mlp=r3(rec["mlp_write"].norm(dim=-1)),
        stay=r3(stay), move=[r3(m) for m in move],
        heads=[[r3(hq) for hq in hb] for hb in heads],
        output=final,
        contributions=rec["value_norms"] is not None,
        pca=dict(all=layer_pca(rec["h"]), skip_first=layer_pca(rec["h"], skip_first=True)),
    )
