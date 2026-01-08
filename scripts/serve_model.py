#!/usr/bin/env python
"""Start an ASR inference server for a specified model.

Usage:
    python -m scripts.serve_model --model whisper --port 8000
    python -m scripts.serve_model --model whisper --model-size large-v3 --port 8000
    python -m scripts.serve_model --model whisper --param compute_type=float16
"""

import argparse

from src.models import MODEL_REGISTRY, get_model_class, get_model_defaults


def parse_param(param: str) -> tuple[str, str | int | float | bool]:
    """Parse a key=value parameter string into a tuple.

    Attempts to convert value to appropriate type (int, float, bool, or str).
    """
    if "=" not in param:
        raise argparse.ArgumentTypeError(
            f"Invalid parameter format: '{param}'. Expected key=value"
        )

    key, value = param.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        raise argparse.ArgumentTypeError(f"Empty key in parameter: '{param}'")

    # Try to convert value to appropriate type
    # Check for boolean
    if value.lower() in ("true", "yes", "1"):
        return key, True
    if value.lower() in ("false", "no", "0"):
        return key, False

    # Try int
    try:
        return key, int(value)
    except ValueError:
        pass

    # Try float
    try:
        return key, float(value)
    except ValueError:
        pass

    # Keep as string
    return key, value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start an ASR inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start Whisper server with default settings (base model)
    python -m scripts.serve_model --model whisper

    # Start Whisper server with large-v3 model
    python -m scripts.serve_model --model whisper --model-size large-v3

    # Start Whisper with custom compute type
    python -m scripts.serve_model --model whisper --param compute_type=int8

    # Start Parakeet server
    python -m scripts.serve_model --model parakeet

    # Multiple model-specific params
    python -m scripts.serve_model --model whisper --param beam_size=10 --param language=en
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
        help="Model size/variant (e.g., 'base', 'large-v3' for Whisper, 'nvidia/parakeet-ctc-1.1b' for Parakeet)",
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
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: from model defaults)",
    )
    parser.add_argument(
        "--param",
        type=parse_param,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Model-specific parameter (can be used multiple times). "
             "E.g., --param compute_type=float16 --param beam_size=5",
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

    # Apply model-specific params from --param arguments
    for key, value in args.param:
        model_kwargs[key] = value

    print(f"Initializing {args.model} model with config:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")
    print()

    # Create and serve model
    model = model_class(**model_kwargs)
    model.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
