"""Finding a model's transformer blocks, reading their outputs, and naming tokens."""

from __future__ import annotations

import re
from typing import Sequence

import torch
from torch import nn

# Substrings that mark a ModuleList as belonging to a non-text tower
# (e.g. the vision encoder of Gemma-3 or LLaVA). These are skipped when
# auto-detecting the decoder blocks.
_NON_TEXT_HINTS = ("vision", "visual", "image", "audio", "encoder")


def find_blocks(model: nn.Module, path: str | None = None) -> nn.ModuleList:
    """Return the ``ModuleList`` holding the model's transformer blocks.

    Args:
        model: A Hugging Face (or any PyTorch) language model.
        path: Optional dotted attribute path, e.g. ``"model.layers"`` or
            ``"transformer.h"``. When omitted, the longest ``ModuleList`` that is
            not part of a vision/audio tower is used, which covers GPT-2, Llama,
            Mistral, Qwen, Gemma (including multimodal Gemma-3), Phi, GPT-NeoX
            and most other decoder-only architectures.
    """
    if path is not None:
        module = model.get_submodule(path)
        if not isinstance(module, nn.ModuleList):
            raise TypeError(f"{path!r} is a {type(module).__name__}, not an nn.ModuleList")
        return module

    candidates = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.ModuleList) and len(m) > 0
    ]
    if not candidates:
        raise ValueError("Could not find any nn.ModuleList of transformer blocks; pass `layers_path`.")

    text_only = [(n, m) for n, m in candidates if not any(h in n.lower() for h in _NON_TEXT_HINTS)]
    pool = text_only or candidates
    # Prefer lists whose children all share a type (a stack of identical blocks),
    # then the longest such stack.
    _, blocks = max(pool, key=lambda nm: (len({type(c) for c in nm[1]}) == 1, len(nm[1])))
    return blocks


def _first_tensor(x) -> torch.Tensor:
    """Blocks return a tensor, a tuple whose first item is the hidden state, or a ModelOutput."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return _first_tensor(x[0])
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    raise TypeError(f"Don't know how to read hidden states from a {type(x).__name__}")


def _attention_projections(block: nn.Module):
    """Find a block's value projection and attention output projection.

    Returns ``(value_module, take_values, out_weight)`` where ``take_values``
    picks the value part out of ``value_module``'s output, and ``out_weight`` is
    the output projection as an ``(n_heads * head_dim, d_model)`` matrix. Returns
    ``None`` for layouts this doesn't recognize.
    """
    for _, mod in block.named_modules():
        # Llama, Mistral, Qwen, Gemma, ...: separate q/k/v projections, nn.Linear weights are (out, in).
        if hasattr(mod, "v_proj") and hasattr(mod, "o_proj"):
            return mod.v_proj, (lambda out: out), lambda: mod.o_proj.weight.T
        # GPT-2: fused qkv in c_attn; Conv1D weights are already (in, out).
        if hasattr(mod, "c_attn") and hasattr(mod, "c_proj"):
            return mod.c_attn, (lambda out: out.chunk(3, dim=-1)[2]), lambda: mod.c_proj.weight
    return None


def head_writes(values: torch.Tensor, out_weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    """What each token writes into the residual stream through each head, per unit of attention.

    Args:
        values: ``(n_tokens, n_kv_heads * head_dim)`` value-projection output.
        out_weight: ``(n_heads * head_dim, d_model)`` attention output projection.
        n_heads: Number of query heads. With grouped-query attention each value
            head is shared by ``n_heads // n_kv_heads`` consecutive query heads.

    Returns:
        ``(n_heads, n_tokens, d_model)``. The head's output for query ``i`` is
        ``sum_j attention[h, i, j] * writes[h, j]`` (before the output bias).
    """
    head_dim = out_weight.shape[0] // n_heads
    n_kv = values.shape[-1] // head_dim
    v = values.view(values.shape[0], n_kv, head_dim).repeat_interleave(n_heads // n_kv, dim=1)
    return torch.einsum("shd,hde->hse", v, out_weight.view(n_heads, head_dim, -1))


_BYTE_CHARS = None


def _gpt2_byte_chars() -> dict[str, int]:
    """GPT-2-style byte-level BPE writes each byte as a printable character; this maps them back.

    Printable Latin-1 bytes stand for themselves; the rest are shifted to code points 256 and up.
    """
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[chr(b)] = b
        else:
            chars[chr(256 + extra)] = b
            extra += 1
    return chars


def _token_bytes(tokenizer, token_id: int) -> bytes | None:
    """The raw bytes of a byte-level token, or None if they can't be recovered."""
    global _BYTE_CHARS
    piece = tokenizer.convert_ids_to_tokens(token_id)
    m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece or "")
    if m:  # SentencePiece byte fallback (Gemma, Llama)
        return bytes([int(m.group(1), 16)])
    if _BYTE_CHARS is None:
        _BYTE_CHARS = _gpt2_byte_chars()
    try:
        return bytes(_BYTE_CHARS[c] for c in piece)
    except (KeyError, TypeError):
        return None


def token_str(tokenizer, token_id: int) -> str:
    """One token as text. A token that holds only part of a character (so it would
    decode to "�") is shown by its bytes instead, e.g. ``"⟨e6 97⟩"``; a leading
    space byte stays a space."""
    text = tokenizer.decode([token_id])
    if "\ufffd" not in text:
        return text
    try:
        raw = _token_bytes(tokenizer, token_id)
    except Exception:
        raw = None
    if not raw:
        return text
    lead = " " if raw[:1] == b" " else ""
    raw = raw[1:] if lead else raw
    return lead + "⟨" + " ".join(f"{b:02x}" for b in raw) + "⟩"


def decode_tokens(tokenizer, input_ids: Sequence[int]) -> list[str]:
    """Decode each token id on its own, so each grid column is one token."""
    return [token_str(tokenizer, i) for i in input_ids]
