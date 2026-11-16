"""The Jacobian lens (J-lens) from "Verbalizable Representations Form a Global
Workspace in Language Models" (Anthropic, 2026):
https://transformer-circuits.pub/2026/workspace/index.html

The J-lens reads out which tokens a residual stream vector is disposed to make
the model say. It carries the vector from its layer to the last block's output
with an average Jacobian, then decodes with the model's own final norm and
unembedding::

    lens_l(h) = softmax(W_U · norm(J_l h)),    J_l = E[∂h_final / ∂h_l]

The Jacobian estimator follows Anthropic's reference implementation
(https://github.com/anthropics/jacobian-lens, Apache-2.0), so lenses fitted here
and the pre-fitted ones Neuronpedia publishes are interchangeable.

Layer numbering: residual stream indices, where 0 is the embeddings and ``i`` is
the output of block ``i - 1``. A lens fitted on a model with ``n`` blocks covers
indices ``1 .. n``; index ``n`` (the last block's output) needs no transport, so
its lens is the model's own output.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .blocks import find_blocks

#: Positions before this are left out of the Jacobian average: early positions act
#: as attention sinks and have atypical residual statistics (reference default).
SKIP_FIRST_N_POSITIONS = 16

#: Pre-fitted lenses published by Neuronpedia (fitted on wikitext-103 with the
#: reference code), keyed by Hugging Face model id.
_NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
_PRETRAINED = {
    "gpt2": "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
    "openai-community/gpt2": "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
    **{
        f"google/gemma-3-{size}": f"gemma-3-{size.removesuffix('-pt')}/jlens/Salesforce-wikitext/"
        f"gemma-3-{size}_jacobian_lens.pt"
        for size in ("270m", "1b-pt", "4b-pt", "12b-pt", "27b-pt")
    },
    **{
        f"google/gemma-3-{size}-it": f"gemma-3-{size}-it/jlens/Salesforce-wikitext/"
        f"gemma-3-{size}-it_jacobian_lens.pt"
        for size in ("270m", "1b", "4b", "12b", "27b")
    },
}


@dataclass
class JacobianLens:
    """Per-block average Jacobians ``J_l``, each ``(d_model, d_model)``.

    Attributes:
        jacobians: ``{block_index: J}``. ``J`` maps the output of block
            ``block_index`` to the output of the last block. Keys use block
            indices, as in the reference code and the published files.
        n_prompts: Number of prompts the average was taken over.
        d_model: Residual stream width.
    """

    jacobians: dict[int, torch.Tensor]
    n_prompts: int
    d_model: int

    def __post_init__(self):
        self.jacobians = {int(k): v.float() for k, v in self.jacobians.items()}

    def __repr__(self) -> str:
        blocks = sorted(self.jacobians)
        return f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, blocks={blocks[0]}..{blocks[-1]})"

    def matrix(self, layer: int) -> torch.Tensor | None:
        """``J`` for residual stream index ``layer``, or ``None`` for the identity (last block)."""
        block = layer - 1
        if block in self.jacobians:
            return self.jacobians[block]
        if block == max(self.jacobians) + 1:
            return None
        raise KeyError(f"No Jacobian for residual stream layer {layer} (fitted blocks: {sorted(self.jacobians)})")

    @property
    def layers(self) -> list[int]:
        """Residual stream indices this lens can read out, including the final one."""
        return [b + 1 for b in sorted(self.jacobians)] + [max(self.jacobians) + 2]

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save in the reference implementation's file format."""
        torch.save(
            {
                "J": {b: J.to(dtype) for b, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": sorted(self.jacobians),
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "JacobianLens":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in ck:
            raise ValueError(f"{path} is not a Jacobian lens file (keys: {sorted(ck)})")
        return cls(jacobians=ck["J"], n_prompts=ck["n_prompts"], d_model=ck["d_model"])

    @classmethod
    def from_pretrained(
        cls, name: str, *, repo: str = _NEURONPEDIA_REPO, filename: str | None = None, revision: str = "main"
    ) -> "JacobianLens":
        """Load a lens from a local file or the Hugging Face Hub.

        Args:
            name: A local ``.pt`` path, or a model id with a published lens
                (``"gpt2"``, ``"google/gemma-3-1b-pt"``, ...; see ``PRETRAINED_LENSES``).
            repo, filename, revision: Override where on the Hub to look.
        """
        if os.path.isfile(name):
            return cls.load(name)
        if filename is None:
            if name not in _PRETRAINED:
                raise KeyError(
                    f"No known published lens for {name!r}. Known: {sorted(_PRETRAINED)}. "
                    "Pass `filename=` (and `repo=`), or fit one with `fit_lens`."
                )
            filename = _PRETRAINED[name]
        from huggingface_hub import hf_hub_download

        return cls.load(hf_hub_download(repo, filename, revision=revision))


PRETRAINED_LENSES = sorted(_PRETRAINED)


def _block_output(output) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


def fit_lens(
    model: nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    dim_batch: int = 32,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    layers_path: str | None = None,
    progress: bool = True,
) -> JacobianLens:
    """Fit a Jacobian lens: ``J_l = E[∂h_final / ∂h_l]`` for every block ``l`` before the last.

    Same estimator as the reference implementation. For each prompt and each
    output dimension, a one-hot gradient is placed at every valid target
    position at once and backpropagated, so the gradient at source position
    ``p`` is ``sum_{p' >= p} ∂h_final[p'] / ∂h_l[p]``; this is averaged over valid
    source positions, then over prompts. Valid positions skip the first
    ``skip_first`` (attention-sink) positions and the last one.

    Cost per prompt: one forward pass and ``ceil(d_model / dim_batch)`` backward
    passes. The paper uses 1,000 prompts of 128 tokens from web text; ~100 is
    usable, and published lenses exist for some models (``JacobianLens.from_pretrained``).

    Args:
        model, tokenizer: The model to fit on.
        prompts: Texts to average over. Ones with fewer than ``skip_first + 2``
            tokens are skipped.
        dim_batch: Output dimensions per backward pass (memory knob).
        max_seq_len: Truncate prompts to this many tokens.
        skip_first: Leading positions left out of the average.
        layers_path: Dotted path to the blocks if auto-detection picks the wrong list.
        progress: Print one line per prompt.
    """
    blocks = find_blocks(model, layers_path)
    n_blocks = len(blocks)
    sources = list(range(n_blocks - 1))
    device = next(model.parameters()).device
    d_model = None
    total: dict[int, torch.Tensor] = {}
    n_done = 0

    recorded: dict[int, torch.Tensor] = {}
    handles = [
        b.register_forward_hook(lambda m, a, o, i=i: recorded.__setitem__(i, _block_output(o)))
        for i, b in enumerate(blocks)
    ]
    requires_grad = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    try:
        model.eval()
        for p in model.parameters():  # only gradients w.r.t. activations are needed
            p.requires_grad_(False)

        # The first source block's output becomes the root of the graph.
        def make_root(m, a, o):
            _block_output(o).requires_grad_(True)

        handles.append(blocks[0].register_forward_hook(make_root))

        for k, prompt in enumerate(prompts):
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)["input_ids"]
            seq = ids.shape[1]
            valid = torch.arange(skip_first, seq - 1)
            if len(valid) == 0:
                if progress:
                    print(f"  skipping prompt {k}: only {seq} tokens")
                continue
            with torch.enable_grad():
                recorded.clear()
                model(input_ids=ids.to(device).expand(dim_batch, -1), use_cache=False)
                target = recorded[n_blocks - 1]
                inputs = [recorded[i] for i in sources]
                d_model = target.shape[-1]
                per_prompt = {i: torch.zeros(d_model, d_model) for i in sources}
                cotangent = torch.zeros_like(target)
                rows = torch.arange(dim_batch, device=target.device)
                vpos = valid.to(target.device)
                n_passes = math.ceil(d_model / dim_batch)
                for pass_idx, start in enumerate(range(0, d_model, dim_batch)):
                    n = min(dim_batch, d_model - start)
                    cotangent.zero_()
                    cotangent[rows[:n, None], vpos[None, :], start + rows[:n, None]] = 1.0
                    grads = torch.autograd.grad(
                        target, inputs, grad_outputs=cotangent, retain_graph=pass_idx < n_passes - 1
                    )
                    for i, g in zip(sources, grads):
                        per_prompt[i][start : start + n] = g[:n, vpos].float().mean(dim=1).cpu()
            for i in sources:
                total[i] = total.get(i, 0) + per_prompt[i]
            n_done += 1
            if progress:
                print(f"  prompt {k + 1}/{len(prompts)}: {seq} tokens")
    finally:
        for h in handles:
            h.remove()
        for p, rg in zip(model.parameters(), requires_grad):
            p.requires_grad_(rg)
        model.train(was_training)

    if n_done == 0:
        raise ValueError(f"No prompt had more than {skip_first + 1} tokens.")
    return JacobianLens({i: J / n_done for i, J in total.items()}, n_prompts=n_done, d_model=d_model)


def _output_head(model: nn.Module):
    """``(final_norm, unembedding_weight, logit_softcap)`` for a causal LM."""
    for path in ("model.norm", "model.language_model.norm", "language_model.norm",
                 "model.final_layernorm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        try:
            norm = model.get_submodule(path)
            break
        except AttributeError:
            continue
    else:
        raise ValueError(f"Could not find the final norm of {type(model).__name__}")
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return norm, model.get_output_embeddings().weight, getattr(config, "final_logit_softcapping", None)


@torch.no_grad()
def lens_logits(model: nn.Module, lens: JacobianLens | None, h: torch.Tensor, layer: int, *,
                on_device: bool = False) -> torch.Tensor:
    """Vocabulary logits the lens reads from residual vectors ``h`` (``(..., d_model)``) at ``layer``.

    ``lens=None`` gives the logit lens (no transport), the usual baseline.
    ``on_device=True`` leaves the result on the model's device, which saves a large
    copy when only the top words or one word's rank are needed.
    """
    norm, w_u, softcap = _output_head(model)
    J = lens.matrix(layer) if lens is not None else None
    x = h.float() if J is None else h.float() @ J.T
    logits = (norm(x.to(w_u.device, w_u.dtype)) @ w_u.T).float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits if on_device else logits.cpu()


def _clean(token: str) -> str:
    return token.replace("\n", "⏎").replace("\t", "⇥")
