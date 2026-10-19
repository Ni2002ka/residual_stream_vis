"""Command line: ``residual-stream-vis gpt2`` opens the flow viewer in your browser."""

from __future__ import annotations

import argparse

from .lens import JacobianLens


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="residual-stream-vis",
        description="Open a local page where you type prompts and trace them through a language model.",
    )
    parser.add_argument("model", help="Hugging Face model id or local path, e.g. gpt2 or google/gemma-3-1b-pt")
    parser.add_argument("examples", nargs="*", help="Prompts to offer as one-click examples")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="Where to listen (default: this machine only)")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: picked automatically)")
    parser.add_argument("--lens", default="auto",
                        help="A J-lens file (.pt) to use, 'none' for the logit lens only, or 'auto' (default) "
                             "to load the published lens when there is one")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)  # old spelling; serving is the default now
    args = parser.parse_args(argv)

    from .server import serve

    lens = {"auto": "auto", "none": None}.get(args.lens.lower(), None)
    if args.lens.lower() not in ("auto", "none"):
        lens = JacobianLens.load(args.lens)
    serve(args.model, lens=lens, host=args.host, port=args.port, device=args.device,
          examples=args.examples or None, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
