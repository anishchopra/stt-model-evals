#!/usr/bin/env python
"""Start an ASR inference server for a specified model.

Usage:
    uv run python -m scripts.serve_model --model whisper --port 8000
    uv run python -m scripts.serve_model --model whisper --model-size large-v3 --port 8000
"""

import argparse

from src.models import MODEL_REGISTRY, get_model_class, get_model_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start an ASR inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start Whisper server with default settings (base model)
    uv run python scripts/serve_model.py --model whisper

    # Start Whisper server with large-v3 model
    uv run python scripts/serve_model.py --model whisper --model-size large-v3

    # Start on specific port
    uv run python scripts/serve_model.py --model whisper --port 8001
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to serve",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Model size/variant (e.g., 'base', 'small', 'large-v3' for Whisper)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        choices=["float16", "int8", "int8_float16", "float32"],
        help="Compute type for inference (default: float16)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Get model class and defaults
    model_class = get_model_class(args.model)
    model_kwargs = get_model_defaults(args.model)

    # Override defaults with CLI args
    if args.model_size:
        model_kwargs["model_name"] = args.model_size
    if args.device:
        model_kwargs["device"] = args.device
    if args.compute_type:
        model_kwargs["compute_type"] = args.compute_type

    print(f"Initializing {args.model} model with config:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")
    print()

    # Create and serve model
    model = model_class(**model_kwargs)
    model.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
