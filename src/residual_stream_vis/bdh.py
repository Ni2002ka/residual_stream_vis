"""The Baby Dragon Hatchling (BDH) architecture, laid out so the viewer can trace it.

BDH (Kosowski et al., 2025, https://arxiv.org/abs/2509.26507) replaces the
Transformer block with a layer of sparse, positive neurons. The math here is
Pathway's reference implementation (https://github.com/pathwaycom/bdh, MIT),
unchanged; only the layout differs, so the viewer's hooks find what they need.
Each layer, with ``x`` the residual stream (``(T, D)``, LayerNormed)::

    x_sparse = relu(x @ E)                       # (heads, T, N): sparse neuron activations
    a        = (rope(x_sparse) @ rope(x_sparse)^T).tril(-1)   # linear attention, strictly causal
    y_kv     = ln(a @ x)                          # what each head read from earlier tokens
    y_sparse = relu(y_kv @ E_v)                   # ... as neurons
    y        = ln((x_sparse * y_sparse) @ D)      # neurons that fire on both write back
    x        = ln(x + y)

The encoders ``E``, ``E_v`` and decoder ``D`` are shared by every layer, there
are no softmax and no separate MLP, and a token can't attend to itself.

For the viewer:

* **Blocks.** ``model.model.layers`` holds one module per layer; they all use
  the same weights.
* **Attention.** BDH's scores are unnormalized and can be negative. The viewer
  gets each query's ``|a|`` scaled to sum to 1 as its "attention". Every
  value vector is a LayerNormed ``x``, the same size for every token, so
  ``|a|`` alone gives each token's share.
* **Writes.** ``ln(x + y)`` splits exactly into a rescaled ``x`` and a
  rescaled, centered ``y``. ``y`` is the only part that depends on other
  tokens (with nothing to attend to, ``y = 0``), so it is reported as the
  attention write; the rest, the layer norm shrinking or growing ``x``, takes
  the MLP's slot.
* **Lens.** The final norm is the same parameter-free LayerNorm; the model's
  own output skips it, but its input is already normalized, so it changes
  nothing.

The tokenizer is bytes (vocabulary 256), as in the reference training script.
Train a small model on tiny Shakespeare with ``python -m residual_stream_vis.bdh``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

#: Prompts for a model trained on tiny Shakespeare.
EXAMPLES = [
    "ROMEO:\nBut, soft! what light through yonder window",
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou",
    "KING RICHARD III:\nA horse! a horse! my kingdom for a",
    "JULIET:\nO Romeo, Romeo! wherefore art thou",
]

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclasses.dataclass
class BDHConfig:
    """Same fields and defaults as the reference ``BDHConfig``."""

    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256

    @property
    def n_neurons(self) -> int:
        """Neurons per head, ``N``."""
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return 1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi)


def rope(phases, v):
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    phases = (phases % 1) * (2 * math.pi)
    return (v * torch.cos(phases)).to(v.dtype) + (v_rot * torch.sin(phases)).to(v.dtype)


class Tap(nn.Identity):
    """Passes its input through; a place for hooks to read a value from."""


class BDHLayer(nn.Module):
    """One BDH layer. Its weights belong to the shared :class:`BDHModel`."""

    def __init__(self, shared: "BDHModel", index: int):
        super().__init__()
        self.__dict__["shared"] = shared  # a plain attribute, so the weights aren't registered (and saved) once per layer
        self.index = index
        self.write = Tap()       # the attention-dependent part of the update (see the module docstring)
        self.neurons = Tap()     # x_sparse * y_sparse, (heads, T, N)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False):
        m, cfg = self.shared, self.shared.config
        B, T, D = hidden_states.shape
        x = hidden_states.unsqueeze(1)                               # B, 1, T, D
        x_sparse = F.relu(x @ m.encoder)                             # B, nh, T, N
        phases = torch.arange(T, device=m.freqs.device, dtype=m.freqs.dtype).view(1, 1, -1, 1) * m.freqs
        qr = rope(phases, x_sparse)
        scores = (qr @ qr.mT).tril(diagonal=-1)                      # B, nh, T, T
        y_kv = m.norm(scores @ x)
        y_sparse = F.relu(y_kv @ m.encoder_v)
        xy_sparse = m.drop(self.neurons(x_sparse * y_sparse))
        y = m.norm(xy_sparse.transpose(1, 2).reshape(B, 1, T, -1) @ m.decoder)
        out = m.norm(x + y)
        # out = (x + y - mean) / s: the y part is this layer's attention-dependent write
        s = (x + y).var(dim=-1, unbiased=False, keepdim=True).add(m.norm.eps).sqrt()
        self.write(((y - y.mean(dim=-1, keepdim=True)) / s).squeeze(1))
        out = out.squeeze(1)
        if not output_attentions:
            return out
        a = scores.abs()
        return out, a / a.sum(-1, keepdim=True).clamp_min(1e-12)


class BDHModel(nn.Module):
    """Embedding, the shared weights, and the layers that use them."""

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D, N = config.n_head, config.n_embd, config.n_neurons
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.freqs = nn.Buffer(get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))
        self.norm = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(BDHLayer(self, i) for i in range(config.n_layer))


class BDHForCausalLM(nn.Module):
    """BDH with the parts of the Hugging Face causal-LM interface the viewer uses."""

    default_examples = EXAMPLES
    update_parts = dict(  # how the viewer labels a layer's update (see the module docstring)
        attn="+ neuron write", mlp="+ norm rescale",
        note="Vector sizes. BDH has no separate MLP: neurons that fire both for this token and for what attention "
             "read from earlier tokens write the update, then a layer norm rescales the whole state.")

    def __init__(self, config: BDHConfig, name: str | None = None):
        super().__init__()
        self.model = BDHModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self.config = SimpleNamespace(**dataclasses.asdict(config), model_type="bdh", _name_or_path=name,
                                      _attn_implementation="eager", num_hidden_layers=config.n_layer)
        self.bdh_config = config

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_attn_implementation(self, impl: str) -> None:
        pass  # BDH has one attention; the scores always come out

    def forward(self, input_ids, attention_mask=None, labels=None, output_attentions: bool = False, **kwargs):
        x = self.model.norm(self.model.embed(input_ids))
        attentions = []
        for layer in self.model.layers:
            out = layer(x, output_attentions=output_attentions)
            if output_attentions:
                x, a = out
                attentions.append(a)
            else:
                x = out
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return SimpleNamespace(logits=logits, loss=loss, attentions=tuple(attentions) if output_attentions else None)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 100, do_sample: bool = False,
                 temperature: float = 1.0, top_k: int | None = None, **kwargs) -> torch.Tensor:
        """Greedy (or sampled) continuation; BDH has no cache, so each step reruns the whole sequence."""
        idx = input_ids
        for _ in range(max_new_tokens):
            logits = self(idx).logits[:, -1, :]
            if do_sample:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    def save(self, path: str) -> None:
        torch.save({"format": "residual_stream_vis.bdh", "config": dataclasses.asdict(self.bdh_config),
                    "state_dict": self.state_dict()}, path)


def from_reference_state_dict(state_dict: dict, n_layer: int = 6, dropout: float = 0.0) -> BDHForCausalLM:
    """Load weights saved from Pathway's ``bdh.BDH`` (``torch.save(model.state_dict())``).

    The shapes give everything but the number of layers, which defaults to the reference's 6.
    """
    sd = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}  # torch.compile prefix
    nh, D, N = sd["encoder"].shape
    vocab = sd["lm_head"].shape[1]
    cfg = BDHConfig(n_layer=n_layer, n_embd=D, n_head=nh, mlp_internal_dim_multiplier=N * nh // D,
                    vocab_size=vocab, dropout=dropout)
    model = BDHForCausalLM(cfg)
    model.model.load_state_dict({"decoder": sd["decoder"], "encoder": sd["encoder"], "encoder_v": sd["encoder_v"],
                                 "freqs": sd["attn.freqs"], "embed.weight": sd["embed.weight"]})
    model.lm_head.weight.data.copy_(sd["lm_head"].T)
    return model


def is_checkpoint(path: str) -> bool:
    return os.path.isfile(path) and path.endswith((".pt", ".pth"))


def load(path: str, *, device: str | None = None, n_layer: int = 6):
    """A BDH checkpoint (from :meth:`BDHForCausalLM.save` or the reference code) and a byte tokenizer."""
    from .models import default_device

    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") == "residual_stream_vis.bdh":
        model = BDHForCausalLM(BDHConfig(**ck["config"]), name=path)
        model.load_state_dict(ck["state_dict"])
    else:
        model = from_reference_state_dict(ck.get("model", ck), n_layer=n_layer)
        model.config._name_or_path = path
    model.to(device or default_device()).eval()
    return model, ByteTokenizer()


class ByteTokenizer:
    """UTF-8 bytes as tokens, with the parts of the Hugging Face tokenizer interface the viewer uses."""

    all_special_ids: list[int] = []
    bos_token = bos_token_id = eos_token_id = pad_token_id = None
    chat_template = None
    vocab_size = 256

    def __call__(self, text, return_tensors=None, add_special_tokens=True, truncation=False, max_length=None):
        ids = list(text.encode("utf-8"))
        if truncation and max_length:
            ids = ids[:max_length]
        if return_tensors is None:
            return {"input_ids": ids}
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids, skip_special_tokens=False):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return bytes(ids).decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, i: int) -> str:
        return chr(i) if 32 <= i < 127 else f"<0x{i:02X}>"


# ------------------------------------------------------------------ training


def _shakespeare(path: str) -> bytes:
    if not os.path.exists(path):
        import urllib.request

        print(f"Downloading tiny Shakespeare to {path}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "rb") as f:
        return f.read()


def train(data: bytes, config: BDHConfig, *, iters: int = 3000, batch_size: int = 8, block_size: int = 256,
          lr: float = 1e-3, weight_decay: float = 0.1, device: str | None = None, log_every: int = 100,
          seed: int = 1337) -> BDHForCausalLM:
    """Train BDH on ``data`` (bytes) as the reference script does: AdamW, constant learning rate, random windows."""
    from .models import default_device

    device = device or default_device()
    torch.manual_seed(seed)
    tokens = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    model = BDHForCausalLM(config).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    t0, running = time.time(), []
    for step in range(iters):
        ix = torch.randint(len(tokens) - block_size - 1, (batch_size,))
        x = torch.stack([tokens[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in ix]).to(device)
        loss = model(x, labels=y).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        running.append(loss.item())
        if (step + 1) % log_every == 0 or step == 0:
            print(f"  step {step + 1}/{iters}  loss {sum(running) / len(running):.3f}  ({time.time() - t0:.0f}s)", flush=True)
            running = []
    return model.eval()


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m residual_stream_vis.bdh",
                                description="Train a BDH model on tiny Shakespeare and fit its J-lens.")
    p.add_argument("--out", default="checkpoints/bdh-shakespeare.pt")
    p.add_argument("--data", default="checkpoints/tinyshakespeare.txt", help="Downloaded here if missing")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--multiplier", type=int, default=128, help="Neurons per head = multiplier × n_embd / n_head")
    p.add_argument("--lens-prompts", type=int, default=64, help="Validation windows to fit the J-lens on (0 to skip)")
    p.add_argument("--lens-only", action="store_true", help="Skip training: fit the J-lens for the checkpoint at --out")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    data = _shakespeare(args.data)
    split = int(0.9 * len(data))
    tok = ByteTokenizer()
    if args.lens_only:
        model, _ = load(args.out, device=args.device)
        _fit_and_save_lens(model, tok, data[split:], args.lens_prompts, args.out)
        return
    cfg = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                    mlp_internal_dim_multiplier=args.multiplier)
    print(f"Training BDH ({cfg}) for {args.iters} steps")
    model = train(data[:split], cfg, iters=args.iters, batch_size=args.batch_size, block_size=args.block_size,
                  device=args.device)
    model.config._name_or_path = args.out
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"Saved {args.out}")

    prompt = "ROMEO:\n"
    ids = model.generate(**{k: v.to(next(model.parameters()).device) for k, v in tok(prompt, return_tensors="pt").items()},
                         max_new_tokens=200)
    print("Sample:\n" + tok.decode(ids[0].tolist()))

    if args.lens_prompts:
        _fit_and_save_lens(model, tok, data[split:], args.lens_prompts, args.out)
    print(f"\nOpen the viewer with:  residual-stream-vis {args.out}")


def _fit_and_save_lens(model, tok, val: bytes, n_prompts: int, checkpoint: str) -> None:
    from .lens import fit_lens

    g = torch.Generator().manual_seed(0)
    starts = torch.randint(len(val) - 200, (n_prompts,), generator=g).tolist()
    # whole UTF-8 characters, ~128 bytes each (tiny Shakespeare is ASCII)
    texts = [val[s : s + 128].decode("utf-8", errors="ignore") for s in starts]
    print(f"Fitting the J-lens on {len(texts)} validation windows", flush=True)
    # a small dim_batch: every copy of the window holds n_head × N neurons per layer
    lens = fit_lens(model, tok, texts, max_seq_len=128, dim_batch=8)
    lens.save(lens_path_for(checkpoint))
    print(f"Saved {lens_path_for(checkpoint)}")


def lens_path_for(checkpoint: str) -> str:
    """Where :func:`main` saves a checkpoint's J-lens, and where the viewer looks for it."""
    root, _ = os.path.splitext(checkpoint)
    return root + ".lens.pt"


if __name__ == "__main__":
    main()
