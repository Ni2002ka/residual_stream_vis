"""Edit one cell of the residual stream and let the layers above recompute.

Reading a cell (a lens, J-space) shows what's in it; editing tests whether the
model uses it. An edit replaces or adds to the residual vector at one
(layer, position) during the forward pass. Everything below and to the left of
the edit is unchanged; everything that depends on it is recomputed.

Layers use the package's residual stream indices: 0 is the embeddings and ``i``
is the output of block ``i - 1``, as in the viewer.
Positions count tokens with special tokens such as ``<bos>`` skipped, unless
``skip_special_tokens=False`` (the viewer's own numbering, which keeps them).
Negative positions count from the end.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .blocks import _first_tensor, find_blocks, token_str
from .lens import JacobianLens, _clean


@dataclass
class Edit:
    """One change to the residual stream.

    Attributes:
        layer: Residual stream index (0 = embeddings, ``i`` = output of block ``i - 1``).
        position: Index into the model's input tokens (special tokens included).
        replace: Put this vector in the cell.
        add: Add this vector to the cell (after ``replace``, if both are given).
    """

    layer: int
    position: int
    replace: torch.Tensor | None = None
    add: torch.Tensor | None = None

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        h = h.clone()
        if self.replace is not None:
            h[0, self.position] = self.replace.to(h.device, h.dtype)
        if self.add is not None:
            h[0, self.position] += self.add.to(h.device, h.dtype)
        return h


def _swap_first(x, h):
    """Put ``h`` where a block's (possibly tuple) output keeps its hidden state."""
    if torch.is_tensor(x):
        return h
    if isinstance(x, tuple):
        return (_swap_first(x[0], h),) + tuple(x[1:])
    if isinstance(x, list):
        return [_swap_first(x[0], h)] + list(x[1:])
    raise TypeError(f"Don't know how to edit a {type(x).__name__}")


@contextmanager
def editing(model: nn.Module, edits: Sequence[Edit], *, layers_path: str | None = None):
    """Apply ``edits`` to every forward pass inside the ``with`` block.

    Hooks registered later (for recording) see the edited values, so a stream
    captured inside the block shows the edit itself and its consequences.
    """
    blocks = find_blocks(model, layers_path)
    handles = []
    try:
        for e in edits:
            if not 0 <= e.layer <= len(blocks):
                raise ValueError(f"layer must be in 0..{len(blocks)}, got {e.layer}")
            if e.layer == 0:  # the input to the first block
                def pre(m, args, kwargs, e=e):
                    if args:
                        return (_swap_first(args[0], e.apply(_first_tensor(args[0]))),) + tuple(args[1:]), kwargs
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = e.apply(kwargs["hidden_states"])
                    return args, kwargs
                handles.append(blocks[0].register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(m, args, out, e=e):
                    return _swap_first(out, e.apply(_first_tensor(out)))
                handles.append(blocks[e.layer - 1].register_forward_hook(post))
        yield
    finally:
        for h in handles:
            h.remove()


def input_position(tokenizer, ids: Sequence[int], position: int, *, skip_special_tokens: bool = True) -> int:
    """Turn a position (special tokens skipped, unless told otherwise) into an index into ``ids``."""
    special = set(tokenizer.all_special_ids) if skip_special_tokens else set()
    keep = [j for j, t in enumerate(ids) if t not in special]
    try:
        return keep[position]
    except IndexError:
        raise IndexError(f"position {position} out of range for {len(keep)} tokens") from None


@torch.no_grad()
def _run(model, tokenizer, prompt: str, edits: Sequence[Edit] = (), layers_path=None):
    """Hidden states (embeddings + every block) and final-position probabilities, with edits applied."""
    blocks = find_blocks(model, layers_path)
    device = next(model.parameters()).device
    hidden = {}

    def rec(m, args, kwargs, out, i):
        if i == 0:
            hidden[0] = _first_tensor(args[0] if args else kwargs["hidden_states"])[0].float().cpu()
        hidden[i + 1] = _first_tensor(out)[0].float().cpu()

    with editing(model, edits, layers_path=layers_path):
        hs = [b.register_forward_hook(lambda m, a, k, o, i=i: rec(m, a, k, o, i), with_kwargs=True) for i, b in enumerate(blocks)]
        try:
            enc = tokenizer(prompt, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0, -1].float().cpu()
        finally:
            for h in hs:
                h.remove()
    return torch.stack([hidden[i] for i in range(len(blocks) + 1)]), logits.softmax(-1), enc["input_ids"][0].tolist()


@dataclass
class EditResult:
    """The model's next-word probabilities before and after an edit."""

    tokenizer: object
    before: torch.Tensor
    after: torch.Tensor
    edit: Edit

    def top(self, k: int = 5, which: str = "after") -> list[tuple[str, float]]:
        probs = self.after if which == "after" else self.before
        top = probs.topk(k)
        return [(_clean(self.tokenizer.decode([i])), float(p)) for i, p in zip(top.indices.tolist(), top.values)]

    def prob(self, word: str) -> tuple[float, float]:
        """``(before, after)`` probability of ``word``'s first token, e.g. ``result.prob(" Paris")``."""
        i = self.tokenizer(word, add_special_tokens=False)["input_ids"][0]
        return float(self.before[i]), float(self.after[i])

    def __repr__(self) -> str:
        fmt = lambda xs: ", ".join(f"{t!r} {p:.0%}" for t, p in xs)
        return f"EditResult(layer {self.edit.layer}, position {self.edit.position})\n  before: {fmt(self.top(3, 'before'))}\n  after:  {fmt(self.top(3))}"


def cell(model, tokenizer, prompt: str, layer: int, position: int = -1, *,
         skip_special_tokens: bool = True, layers_path: str | None = None) -> torch.Tensor:
    """The residual vector of one cell, ``(d_model,)``."""
    h, _, ids = _run(model, tokenizer, prompt, layers_path=layers_path)
    return h[layer, input_position(tokenizer, ids, position, skip_special_tokens=skip_special_tokens)]


def run_edit(model, tokenizer, prompt: str, edit: Edit, *, layers_path: str | None = None) -> EditResult:
    """Run ``prompt`` with and without ``edit`` (positions already in input indices)."""
    _, before, _ = _run(model, tokenizer, prompt, layers_path=layers_path)
    _, after, _ = _run(model, tokenizer, prompt, [edit], layers_path=layers_path)
    return EditResult(tokenizer, before, after, edit)


def patch(model, tokenizer, prompt: str, layer: int, position: int = -1, *, source: str | torch.Tensor,
          source_position: int = -1, skip_special_tokens: bool = True, layers_path: str | None = None) -> EditResult:
    """Replace one cell with the same layer's cell from another prompt (or any vector).

    Example::

        r = rsv.patch(model, tok, "The Eiffel Tower is in the city of", 8,
                      source="The Colosseum is in the city of")
        r.prob(" Rome")        # (before, after)

    Args:
        prompt: The prompt to edit.
        layer, position: The cell to replace (residual stream index; position with special tokens skipped).
        source: A donor prompt, whose cell at ``(layer, source_position)`` is copied in, or a vector.
        source_position: Position in the donor prompt.
    """
    vec = source if torch.is_tensor(source) else cell(
        model, tokenizer, source, layer, source_position, skip_special_tokens=skip_special_tokens, layers_path=layers_path)
    ids = tokenizer(prompt)["input_ids"]
    e = Edit(layer, input_position(tokenizer, ids, position, skip_special_tokens=skip_special_tokens), replace=vec)
    return run_edit(model, tokenizer, prompt, e, layers_path=layers_path)


def steer(model, tokenizer, prompt: str, vector: torch.Tensor, layer: int, position: int = -1, *, alpha: float = 1.0,
          skip_special_tokens: bool = True, layers_path: str | None = None) -> EditResult:
    """Add ``alpha * vector`` to one cell, e.g. a :func:`steering_vector` or :func:`word_direction`."""
    ids = tokenizer(prompt)["input_ids"]
    e = Edit(layer, input_position(tokenizer, ids, position, skip_special_tokens=skip_special_tokens), add=alpha * vector)
    return run_edit(model, tokenizer, prompt, e, layers_path=layers_path)


def steering_vector(model, tokenizer, prompts_a: str | Sequence[str], prompts_b: str | Sequence[str], layer: int,
                    position: int = -1, *, skip_special_tokens: bool = True, layers_path: str | None = None) -> torch.Tensor:
    """Mean cell of ``prompts_a`` minus mean cell of ``prompts_b`` at ``(layer, position)``.

    For example, French→Spanish minus French→Chinese translation prompts at the
    answer position gives a direction that switches the output language.
    """
    def mean(ps):
        ps = [ps] if isinstance(ps, str) else list(ps)
        return torch.stack([cell(model, tokenizer, p, layer, position, skip_special_tokens=skip_special_tokens,
                                 layers_path=layers_path) for p in ps]).mean(0)
    return mean(prompts_a) - mean(prompts_b)


def word_token(tokenizer, word: str) -> tuple[int, str]:
    """The single token for ``word``: as typed if it's one token, else with a leading space, else its first token."""
    forms = [word] if word.startswith(" ") else [word, " " + word]
    for f in forms:
        ids = tokenizer(f, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0], _clean(token_str(tokenizer, ids[0]))
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"{word!r} has no tokens")
    return ids[0], _clean(token_str(tokenizer, ids[0]))


def word_direction(model, tokenizer, word: str, layer: int, lens: JacobianLens | None = None, *,
                   kind: str | None = None, center: bool = True) -> torch.Tensor:
    """Unit direction at ``layer`` for ``word``.

    Two kinds, which are good at different edits:

    * ``"unembed"`` (the logit lens's direction): the word's row of the
      unembedding. The residual connections carry it to the top unchanged, so
      adding it is the most direct way to make *this token* output ``word``.
    * ``"jlens"``: the word's J-lens vector, the row of ``W_U J_l``. ``J`` sums
      a cell's effect on the top row at its own position *and every later one*,
      so this is the direction later computation reads as ``word``. It works
      better for changing a concept that other tokens use (e.g. swapping the
      animal in "How many legs does a chicken have?"), and worse for forcing
      this token's own output.

    Args:
        lens: Needed for ``"jlens"``.
        kind: ``"jlens"`` or ``"unembed"``; default ``"jlens"`` when a lens is given.
        center: Subtract the mean unembedding row first (removes the shared
            direction every word has).
    """
    kind = kind or ("jlens" if lens is not None else "unembed")
    if kind not in ("jlens", "unembed"):
        raise ValueError(f"kind must be 'jlens' or 'unembed', got {kind!r}")
    if kind == "jlens" and lens is None:
        raise ValueError("kind='jlens' needs a lens")
    if kind == "jlens" and layer not in lens.layers:
        raise ValueError(f"The J-lens covers layers {lens.layers[0]}..{lens.layers[-1]}, not {layer}")
    i, _ = word_token(tokenizer, word)
    w = model.get_output_embeddings().weight.detach().float().cpu()
    v = w[i] - w.mean(0) if center else w[i].clone()  # rows of W_U J minus their mean = (W_U - mean row) J
    J = lens.matrix(layer) if kind == "jlens" else None
    if J is not None:
        v = v @ J
    return v / v.norm().clamp_min(1e-12)


def swap_vector(h: torch.Tensor, old: torch.Tensor, new: torch.Tensor, strength: float = 2.0) -> torch.Tensor:
    """``h`` with its component along unit ``old`` removed and ``strength × |that component|`` added along unit ``new``."""
    c = (h @ old).abs().clamp_min(1e-3 * h.norm())
    return h - (h @ old) * old + strength * c * new


def swap(model, tokenizer, prompt: str, old: str, new: str, layer: int, position: int = -1, *,
         lens: JacobianLens | None = None, kind: str | None = None, strength: float = 2.0,
         skip_special_tokens: bool = True, layers_path: str | None = None) -> EditResult:
    """Swap one word's direction for another's in one cell, e.g. the animal in a question.

    Example (Gemma-3-1B, few-shot "How many legs does a spider have?")::

        rsv.swap(model, tok, prompt, "spider", "ant", layer=8, position=spider_pos, lens=lens)
        # answer goes from "8" to "6"; with kind="unembed" it doesn't change

    Uses uncentered directions, as in ``experiments/animal_swap.py``.
    """
    ids = tokenizer(prompt)["input_ids"]
    pos = input_position(tokenizer, ids, position, skip_special_tokens=skip_special_tokens)
    h, _, _ = _run(model, tokenizer, prompt, layers_path=layers_path)
    d = lambda w: word_direction(model, tokenizer, w, layer, lens, kind=kind, center=False)
    e = Edit(layer, pos, replace=swap_vector(h[layer, pos], d(old), d(new), strength))
    return run_edit(model, tokenizer, prompt, e, layers_path=layers_path)
