#!/usr/bin/env python
"""Compute evaluation metrics for a completed inference run.

Usage:
    uv run python -m scripts.compute_metrics --run-name whisper
    uv run python -m scripts.compute_metrics --run-name whisper --output-dir outputs
"""

import argparse
import json
from pathlib import Path

from src.data_loader import load_eval_dataset
from src.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute metrics for an inference run",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of the run (directory name under outputs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)",
    )

    return parser.parse_args()


def load_predictions(run_dir: Path) -> list[dict]:
    """Load predictions from a run directory."""
    predictions_file = run_dir / "predictions.jsonl"
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    predictions = []
    with open(predictions_file) as f:
        for line in f:
            predictions.append(json.loads(line))

    return predictions


def load_metadata(run_dir: Path) -> dict:
    """Load metadata from a run directory."""
    metadata_file = run_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file) as f:
        return json.load(f)


def build_references(dataset) -> dict[str, dict]:
    """Build references dict from dataset."""
    references = {}
    for sample in dataset:
        references[sample.id] = {
            "text": sample.reference_text,
            "emotion": sample.emotion,
        }
    return references


def main():
    args = parse_args()

    # Find run directory
    run_dir = Path(args.output_dir) / args.run_name
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return 1

    print(f"Computing metrics for run: {args.run_name}")
    print(f"  Run directory: {run_dir}")

    # Load metadata
    metadata = load_metadata(run_dir)
    print(f"  Model: {metadata['model']}")
    print(f"  Dataset: {metadata['dataset']}")

    # Load predictions
    predictions = load_predictions(run_dir)
    print(f"  Predictions loaded: {len(predictions)}")

    # Load dataset for references
    dataset = load_eval_dataset(
        data_dir=metadata.get("data_dir", "data"),
        dataset_name=metadata["dataset"],
    )
    references = build_references(dataset)
    print(f"  References loaded: {len(references)}")

    # Compute all metrics
    print("\nComputing metrics...")
    results = compute_all_metrics(predictions, references)

    # Build output dicts
    metrics_summary = {name: result.summary() for name, result in results.items()}
    metrics_per_sample = {name: result.per_sample for name, result in results.items()}

    # Save aggregate metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved: {metrics_file}")

    # Save per-sample metrics
    per_sample_file = run_dir / "metrics_per_sample.json"
    with open(per_sample_file, "w") as f:
        json.dump(metrics_per_sample, f, indent=2)
    print(f"Saved: {per_sample_file}")

    # Print summary
    print(f"\n{'='*50}")
    print("Metrics Summary")
    print(f"{'='*50}")

    for name, result in results.items():
        print(f"\n  [{name.upper()}]")
        for key, value in result.details.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

    return 0


if __name__ == "__main__":
    exit(main())
