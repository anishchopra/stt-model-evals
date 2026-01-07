#!/usr/bin/env python
"""Run inference on the evaluation dataset using a model server.

Usage:
    # First start the server in another terminal:
    uv run -m python scripts.serve_model --model whisper --port 8000

    # Then run inference:
    uv run -m python scripts.run_inference --model whisper --port 8000

    # Run on subset for testing:
    uv run -m python scripts.run_inference --model whisper --port 8000 --limit 10
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.data_loader import load_eval_dataset
from src.models import MODEL_REGISTRY, get_model_class


def get_unique_run_dir(base_dir: Path, run_name: str) -> Path:
    """Get a unique run directory, appending numbers if needed.

    If outputs/whisper exists, tries whisper0, whisper1, etc.
    """
    run_dir = base_dir / run_name
    if not run_dir.exists():
        return run_dir

    # Directory exists, find next available number
    i = 0
    while True:
        run_dir = base_dir / f"{run_name}{i}"
        if not run_dir.exists():
            return run_dir
        i += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on evaluation dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to use",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: model name). Used as output directory name.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval_10h",
        help="Dataset name (default: eval_10h)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples that already have predictions",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine run name (default to model name)
    run_name = args.run_name or args.model

    # Setup output directory with auto-incrementing if exists
    base_dir = Path(args.output_dir)
    output_dir = get_unique_run_dir(base_dir, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    predictions_file = output_dir / "predictions.jsonl"
    metadata_file = output_dir / "metadata.json"

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_eval_dataset(data_dir=args.data_dir, dataset_name=args.dataset)
    print(f"  Total samples: {len(dataset)}")

    if args.limit:
        dataset = dataset.subset(args.limit)
        print(f"  Limited to: {len(dataset)} samples")

    # Load existing predictions if skipping
    existing_ids = set()
    if args.skip_existing and predictions_file.exists():
        with open(predictions_file) as f:
            for line in f:
                data = json.loads(line)
                existing_ids.add(data["id"])
        print(f"  Skipping {len(existing_ids)} existing predictions")

    # Get client
    model_class = get_model_class(args.model)
    model = model_class()
    client = model.get_client(host=args.host, port=args.port)

    # Check server health
    print(f"\nConnecting to server at {args.host}:{args.port}")
    if not client.health_check():
        print("ERROR: Server is not healthy. Make sure it's running:")
        print(f"  uv run python scripts/serve_model.py --model {args.model} --port {args.port}")
        sys.exit(1)
    print("  Server is healthy!")

    # Run inference
    print(f"\nRunning inference...")
    start_time = time.time()
    results = []
    errors = []

    # Open file for streaming writes
    mode = "a" if args.skip_existing else "w"
    with open(predictions_file, mode) as f:
        for sample in tqdm(dataset, desc="Transcribing"):
            if sample.id in existing_ids:
                continue

            try:
                result = client.transcribe(sample.audio_path)

                record = {
                    "id": sample.id,
                    "prediction": result.text,
                    "latency_ms": result.latency_ms,
                    "audio_duration_s": result.audio_duration_s,
                }
                results.append(record)

                # Write immediately (streaming)
                f.write(json.dumps(record) + "\n")
                f.flush()

            except Exception as e:
                error_record = {
                    "id": sample.id,
                    "error": str(e),
                }
                errors.append(error_record)
                tqdm.write(f"Error on {sample.id}: {e}")

    total_time = time.time() - start_time

    # Save metadata
    metadata = {
        "run_name": output_dir.name,
        "model": args.model,
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "host": args.host,
        "port": args.port,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(dataset),
        "successful": len(results),
        "errors": len(errors),
        "total_time_s": total_time,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results) if results else 0,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save errors if any
    if errors:
        errors_file = output_dir / "errors.json"
        with open(errors_file, "w") as f:
            json.dump(errors, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print("Inference Complete!")
    print(f"{'='*50}")
    print(f"  Run: {output_dir.name}")
    print(f"  Successful: {len(results)}/{len(dataset)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Total time: {total_time:.1f}s")
    if results:
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - {predictions_file.name}")
    print(f"  - {metadata_file.name}")
    if errors:
        print(f"  - errors.json")


if __name__ == "__main__":
    main()
