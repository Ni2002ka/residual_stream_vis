"""Trace a prompt through a Hugging Face language model in your browser.

    residual-stream-vis gpt2            # from the shell

    import residual_stream_vis as rsv   # or from Python
    rsv.serve("gpt2")

The page shows the residual stream as a grid (tokens across, layers up), reads
every cell with the J-lens and the logit lens, shows how attention moves
information between tokens, plots each layer's PCA, and lets you edit a cell
and rerun the model from there.

The same pieces work without the page::

    model, tok = rsv.load_model("gpt2")
    lens = rsv.JacobianLens.from_pretrained("gpt2")
    data = rsv.trace(model, tok, "The Eiffel Tower is in the city of", lens)
    rsv.flow_html(data, "flow.html")                       # a static page, no server

    d = rsv.word_direction(model, tok, " London", 8, kind="unembed")   # edit a cell and rerun
    r = rsv.steer(model, tok, prompt, d * rsv.cell(model, tok, prompt, 8).norm(), 8)
    r.prob(" London")                                                   # (before, after)
"""

from . import edit, server
from .blocks import find_blocks
from .edit import Edit, EditResult, cell, editing, patch, steer, steering_vector, swap, word_direction
from .flow import flow_weights, layer_pca, mixing, record, trace
from .lens import PRETRAINED_LENSES, JacobianLens, fit_lens, lens_logits
from .models import load_model
from .server import flow_html, make_server, serve

__version__ = "0.2.0"

__all__ = [
    "Edit",
    "EditResult",
    "JacobianLens",
    "PRETRAINED_LENSES",
    "cell",
    "editing",
    "find_blocks",
    "fit_lens",
    "flow_html",
    "flow_weights",
    "layer_pca",
    "lens_logits",
    "load_model",
    "make_server",
    "mixing",
    "patch",
    "record",
    "serve",
    "steer",
    "steering_vector",
    "swap",
    "trace",
    "word_direction",
]
