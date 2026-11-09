"""A local web app for tracing information flow: type a prompt, see it move through the model.

    import residual_stream_vis as rsv
    rsv.serve("gpt2")             # opens http://127.0.0.1:8000

or from the shell: ``residual-stream-vis gpt2``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Sequence

import torch

from .edit import Edit, cell, swap_vector, word_direction, word_token
from .flow import heads_at, record, trace_record, track
from .lens import JacobianLens, PRETRAINED_LENSES

DEFAULT_EXAMPLES = [
    # Antonyms (the paper's Scenario A). On Gemma-3-4B-it a J-lens swap of undue for rural at strength 4, layers
    # 18-20, gives "urban"; a J-lens swap of comfortable for unfold never gives "fold".
    "Q: sustain\nA: deplete\n\nQ: northeastern\nA: southwestern\n\nQ: supply\nA: demand\n\n"
    "Q: solder\nA: desolder\n\nQ: nervous\nA: calm\n\nQ: undue\nA:",
    "Q: hold\nA: release\n\nQ: bears\nA: cubs\n\nQ: offence\nA: defence\n\n"
    "Q: enemy\nA: friend\n\nQ: packed\nA: empty\n\nQ: comfortable\nA:",
    "The Eiffel Tower is located in the city of",
    "When Mary and John went to the store, John gave a drink to",
    "Hot is the opposite of cold. Up is the opposite of down. Big is the opposite of",
    "Q: How many legs does a frog have?\nA: 4\n\nQ: How many legs does a crab have?\nA: 10\n\n"
    "Q: How many legs does a human have?\nA: 2\n\nQ: How many legs does a spider have?\nA: ",
]


CHAT_EXAMPLES = [  # word answers first: "Track a word" follows them cleanly (Gemma writes numbers digit by digit)
    "The Eiffel Tower is in a country. What is the capital of that country? Think step by step.",
    "Tom's father is Bob. Bob's sister is Alice. Who is Tom's aunt? Think step by step.",
    "I put a red ball inside a blue box, and the box inside a green bag. "
    "What color is the container the ball is directly inside? Think step by step.",
    "Anna is taller than Ben. Ben is taller than Carl. Who is the shortest? Think step by step.",
    "Lisa has 3 boxes with 12 apples in each box. She gives away 7 apples. "
    "How many apples does she have left? Think step by step.",
    "What is 13 + 28? Think step by step.",
]


def _page(config: dict) -> str:
    template = resources.files("residual_stream_vis").joinpath("templates/flow.html").read_text(encoding="utf-8")
    return template.replace("/*RSV_CONFIG*/{}", json.dumps(config, ensure_ascii=False).replace("</", "<\\/"))


def flow_html(data: dict | Sequence[dict], path: str | Path | None = None, *, title: str | None = None) -> str:
    """A standalone flow viewer page for prompts already traced with :func:`trace`.

    Args:
        data: One :func:`trace` result or a list of them (shown as a picker).
        path: Write the page here as well.
        title: Page heading.
    """
    traces = [data] if isinstance(data, dict) else list(data)
    config = {"live": False, "traces": traces, "model": traces[0].get("model"), "title": title}
    page = _page(config)
    if path is not None:
        Path(path).write_text(page, encoding="utf-8")
    return page


def _fit_examples(tokenizer, examples: Sequence[str]) -> list[str]:
    """Adjust example prompts that end in "A: " to the tokenizer.

    Gemma writes " 8" as " " + "8", so the prompt must end with the space for the
    next token to be the digit. GPT-2 writes " 8" as one token, so a prompt ending
    in a bare space is unusual text and it answers with junk like "????".
    """
    space_is_separate = len(tokenizer(" 8", add_special_tokens=False)["input_ids"]) > 1
    return [e if space_is_separate or not e.endswith(": ") else e.rstrip(" ") for e in examples]


DIRECTION_NAMES = {"jlens": "J-lens", "unembed": "logit-lens (unembedding)"}


def _can_stream(model) -> bool:
    """Whether ``model.generate`` takes a Hugging Face ``streamer`` (BDH's own ``generate`` doesn't)."""
    try:
        from transformers import GenerationMixin
    except ImportError:
        return False
    return isinstance(model, GenerationMixin)


def _token_text(t: str) -> str:
    return repr(t.replace("\n", "⏎"))


def _resolve_lens(model, lens):
    if lens != "auto":
        return lens
    from .bdh import lens_path_for  # imported here so `python -m residual_stream_vis.bdh` runs cleanly

    name = getattr(getattr(model, "config", None), "_name_or_path", None)
    if name and os.path.isfile(lens_path_for(name)):  # a local checkpoint with its lens beside it (BDH)
        return JacobianLens.load(lens_path_for(name))
    if name not in PRETRAINED_LENSES:
        print(f"No published J-lens for {name!r}; showing the logit lens only. "
              "Pass lens=rsv.fit_lens(...) to add one.")
        return None
    try:
        return JacobianLens.from_pretrained(name)
    except Exception as e:  # offline, missing file, ...
        print(f"Couldn't load the J-lens for {name!r} ({e}); showing the logit lens only.")
        return None


def make_server(
    model,
    tokenizer=None,
    *,
    lens="auto",
    host: str = "127.0.0.1",
    port: int = 8000,
    examples: Sequence[str] | None = None,
    device: str | None = None,
) -> ThreadingHTTPServer:
    """Build the flow viewer's HTTP server without starting it; see :func:`serve`."""
    if isinstance(model, (str, Path)):
        from .models import load_model
        model, loaded_tok = load_model(str(model), device=device)
        tokenizer = tokenizer or loaded_tok
    elif tokenizer is None:
        raise ValueError("Pass `tokenizer=` when `model` is an already-loaded model.")
    lens = _resolve_lens(model, lens)
    name = getattr(getattr(model, "config", None), "_name_or_path", None)
    chat = bool(getattr(tokenizer, "chat_template", None))
    page = _page({"live": True, "model": name, "examples": _fit_examples(tokenizer, examples or getattr(model, "default_examples", None) or DEFAULT_EXAMPLES),
                  "chat_examples": CHAT_EXAMPLES, "chat": chat, "lens": lens is not None,
                  "neurons": hasattr(model, "neuron_report")})
    device = next(model.parameters()).device
    lock = threading.Lock()
    originals: dict[str, dict] = {}  # prompt -> record of the unedited run (a few, newest last)

    def original(prompt):
        if prompt not in originals:
            originals[prompt] = record(model, tokenizer, prompt)
            while len(originals) > 4:
                originals.pop(next(iter(originals)))
        return originals[prompt]

    edited_last: dict[str, dict] = {}  # prompt -> record of the most recent edited run

    def run_of(prompt: str, edited: bool) -> dict:
        if edited and prompt in edited_last:
            return edited_last[prompt]
        return original(prompt)

    def generate(user_text: str, max_new_tokens: int, use_chat: bool, track_word=None, on_text=None) -> dict:
        """Let the model write a response, then trace prompt + response as one sequence.

        ``on_text(piece)`` is called with the response as it's written: piece by piece
        for Hugging Face models, in one go for models whose ``generate`` can't stream.
        """
        text = user_text
        if use_chat and chat:
            text = tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                                 add_generation_prompt=True)
        bos = getattr(tokenizer, "bos_token", None)
        if bos and text.startswith(bos):  # the tokenizer adds it again
            text = text[len(bos):]
        enc = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None))
        if on_text is not None and _can_stream(model):
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
            result = {}

            def work():
                try:
                    with torch.no_grad():
                        result["out"] = model.generate(**enc, streamer=streamer, **kwargs)
                except BaseException as e:  # unblock the reader, then re-raise below
                    result["error"] = e
                    streamer.end()

            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            for piece in streamer:
                if piece:
                    on_text(piece)
            worker.join()
            if "error" in result:
                raise result["error"]
            out = result["out"]
        else:
            with torch.no_grad():
                out = model.generate(**enc, **kwargs)
        new = out[0, enc["input_ids"].shape[1]:].tolist()
        if on_text is not None and not _can_stream(model):
            on_text(tokenizer.decode(new, skip_special_tokens=True))
        full = text + tokenizer.decode(new, skip_special_tokens=False)
        data = trace_record(original(full), model, tokenizer, full, lens, track_word=track_word)
        data["generated"] = dict(start=int(enc["input_ids"].shape[1]), text=tokenizer.decode(new, skip_special_tokens=True).strip(),
                                 chat=bool(use_chat and chat), user=user_text, n_new=len(new))
        return data

    def direction_kind(spec: dict) -> str:
        dkind = spec.get("direction") or ("jlens" if lens is not None else "unembed")
        if dkind == "jlens" and lens is None:
            raise ValueError("This model has no J-lens loaded; use the logit-lens direction.")
        return dkind

    def run_edit(prompt: str, spec: dict) -> dict:
        base = original(prompt)
        n_layers, n = base["h"].shape[:2]
        layer, pos = int(spec["layer"]), int(spec["position"])
        if not (0 <= layer < n_layers and 0 <= pos < n):
            raise ValueError(f"No cell at layer {layer}, token {pos}.")
        here = base["h"][layer, pos]
        kind, amount = spec.get("kind"), float(spec.get("amount", 1.0))
        tok_text = _token_text(base["tokens"][pos])
        if kind == "patch":
            source = str(spec.get("source", "")).strip()
            if not source:
                raise ValueError("Type a donor prompt to copy the cell from.")
            spos = int(spec.get("source_position", -1))
            vec = cell(model, tokenizer, source, layer, spos, skip_special_tokens=False)
            edit = Edit(layer, pos, replace=vec)
            desc = f"Replaced layer {layer} of {tok_text} with the same layer's last-token cell from “{source}”."
        elif kind == "word":
            word = str(spec.get("word", ""))
            if not word.strip():
                raise ValueError("Type a word to steer toward.")
            dkind = direction_kind(spec)
            _, used = word_token(tokenizer, word)
            vec = word_direction(model, tokenizer, word, layer, lens, kind=dkind) * amount * here.norm()
            edit = Edit(layer, pos, add=vec)
            way = "toward" if amount >= 0 else "away from"
            desc = (f"Pushed layer {layer} of {tok_text} {way} {used!r} by {abs(amount):.2g} × the cell's own size, "
                    f"along the {DIRECTION_NAMES[dkind]} direction.")
        elif kind == "swap":
            old = str(spec.get("old_word", "")).strip() or base["tokens"][pos].strip()
            new = str(spec.get("new_word", "")).strip()
            if not new:
                raise ValueError("Type the word to swap in.")
            dkind = direction_kind(spec)
            d = lambda w: word_direction(model, tokenizer, w, layer, lens, kind=dkind, center=False)
            _, u_old = word_token(tokenizer, old)
            _, u_new = word_token(tokenizer, new)
            strength = float(spec.get("amount", 2.0))
            edit = Edit(layer, pos, replace=swap_vector(here, d(old), d(new), strength))
            desc = (f"Swapped {u_old!r} for {u_new!r} in layer {layer} of {tok_text}, along {DIRECTION_NAMES[dkind]} "
                    f"directions, at strength {strength:g}.")
        elif kind == "diff":
            a, b = str(spec.get("prompt_a", "")).strip(), str(spec.get("prompt_b", "")).strip()
            if not a or not b:
                raise ValueError("Type both prompts for the difference.")
            vec = cell(model, tokenizer, a, layer, -1, skip_special_tokens=False) - cell(model, tokenizer, b, layer, -1, skip_special_tokens=False)
            edit = Edit(layer, pos, add=amount * vec)
            desc = f"Added {amount:.2g} × (last-token cell of “{a}” − “{b}”) at layer {layer} of {tok_text}."
        else:
            raise ValueError(f"Unknown edit kind {kind!r}")
        rec = record(model, tokenizer, prompt, edits=[edit])
        rec["edits"] = [edit]
        edited_last.clear()
        edited_last[prompt] = rec
        data = trace_record(rec, model, tokenizer, prompt, lens, track_word=spec.get("track"))
        change = (rec["h"] - base["h"]).norm(dim=-1) / base["h"].norm(dim=-1).clamp_min(1e-6)
        return dict(trace=data, change=[[round(float(v), 4) for v in row] for row in change],
                    edit=dict(layer=layer, position=pos, kind=kind, description=desc))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, kind: str):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/generate_stream":
                return self.stream_generate()
            if self.path not in ("/api/trace", "/api/edit", "/api/generate", "/api/track", "/api/heads", "/api/neurons"):
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = str(body.get("prompt", ""))  # not stripped: a trailing space changes the next token
                if not prompt.strip():
                    raise ValueError("Type a prompt first.")
                t0 = time.time()
                with lock:
                    if self.path == "/api/trace":
                        data = trace_record(original(prompt), model, tokenizer, prompt, lens, track_word=body.get("track"))
                    elif self.path == "/api/generate":
                        n_new = max(1, min(int(body.get("max_new_tokens", 160)), 512))
                        data = generate(prompt, n_new, bool(body.get("chat", True)), track_word=body.get("track"))
                    elif self.path == "/api/track":
                        word = str(body.get("word", ""))
                        if not word.strip():
                            raise ValueError("Type a word to track.")
                        data = track(run_of(prompt, bool(body.get("edited"))), model, tokenizer, word, lens)
                    elif self.path == "/api/neurons":
                        if not hasattr(model, "neuron_report"):
                            raise ValueError("This model has no neuron view (only BDH models do).")
                        edits = run_of(prompt, bool(body.get("edited"))).get("edits", ())
                        data = model.neuron_report(tokenizer, prompt, int(body["layer"]), int(body["position"]), edits=edits)
                    elif self.path == "/api/heads":
                        rec = run_of(prompt, bool(body.get("edited")))
                        data = {"heads": heads_at(rec, int(body["block"]), int(body["position"]))}
                    else:
                        data = run_edit(prompt, body.get("edit") or {})
                data["seconds"] = round(time.time() - t0, 2)
                self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")

        def stream_generate(self):
            """Server-sent events: ``text`` events as the answer is written, then ``trace`` (or ``error``)."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def event(kind, payload):
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = str(body.get("prompt", ""))
                if not prompt.strip():
                    raise ValueError("Type a prompt first.")
                n_new = max(1, min(int(body.get("max_new_tokens", 160)), 512))
                t0 = time.time()
                with lock:
                    data = generate(prompt, n_new, bool(body.get("chat", True)), track_word=body.get("track"),
                                    on_text=lambda piece: event("text", piece))
                data["seconds"] = round(time.time() - t0, 2)
                event("trace", data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    event("error", str(e))
                except Exception:
                    pass

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.model_name = name
    return server


def serve(
    model,
    tokenizer=None,
    *,
    lens="auto",
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    examples: Sequence[str] | None = None,
    device: str | None = None,
):
    """Start the flow viewer: a page where you type a prompt and trace it through the model.

    Blocks until interrupted (Ctrl-C). Runs on your machine only (``host``
    defaults to localhost).

    Args:
        model: A loaded model, or a Hub id / path to load with :func:`~residual_stream_vis.load_model`.
        tokenizer: Required when ``model`` is already loaded.
        lens: ``"auto"`` loads the published J-lens when one exists, a
            :class:`~residual_stream_vis.JacobianLens`, or ``None`` for the logit lens only.
        host, port: Where to listen.
        open_browser: Open the page in your browser.
        examples: Prompts offered as one-click examples.
        device: Device to load the model on, when ``model`` is a name.
    """
    server = make_server(model, tokenizer, lens=lens, host=host, port=port, examples=examples, device=device)
    name = server.model_name
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Flow viewer for {name} at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
