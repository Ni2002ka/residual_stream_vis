"""Load a Hugging Face causal LM for the viewer."""

from __future__ import annotations

import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, *, device: str | None = None, dtype="auto", **kwargs):
    """Load a Hugging Face causal LM and its tokenizer, ready for the viewer.

    Args:
        name: A Hub id such as ``"gpt2"`` or ``"google/gemma-3-1b-pt"``, or a local path.
        device: ``"cuda"``, ``"mps"`` or ``"cpu"``. Picked automatically if omitted.
        dtype: Passed to ``from_pretrained``; ``"auto"`` uses the checkpoint's dtype.
        **kwargs: Forwarded to ``AutoModelForCausalLM.from_pretrained``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    if "device_map" not in kwargs:  # device_map needs `accelerate`; .to() does not
        model.to(device or default_device())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer
