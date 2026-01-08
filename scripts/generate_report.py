#!/usr/bin/env python
"""Generate comparison reports for multiple inference runs.

Usage:
    python -m scripts.generate_report --runs outputs/whisper outputs/parakeet
    python -m scripts.generate_report --runs outputs/whisper outputs/parakeet --report-name comparison
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.metrics import generate_all_charts


def get_report_dir(base_dir: Path, report_name: str | None) -> Path:
    """Get report directory, auto-generating name if needed.

    If report_name is provided and exists, appends numbers (0, 1, 2, ...).
    If not provided, generates a timestamp-based name.
    """
    if report_name:
        report_dir = base_dir / report_name
        if not report_dir.exists():
            return report_dir

        # Directory exists, find next available number
        i = 0
        while True:
            candidate = base_dir / f"{report_name}{i}"
            if not candidate.exists():
                return candidate
            i += 1

    # Auto-generate timestamp-based name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"report_{timestamp}"


def load_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json from a run directory."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        return json.load(f)


def load_metadata(run_dir: Path) -> dict | None:
    """Load metadata.json from a run directory."""
    metadata_file = run_dir / "metadata.json"
    if not metadata_file.exists():
        return None

    with open(metadata_file) as f:
        return json.load(f)


def collect_run_data(run_paths: list[str]) -> list[dict]:
    """Collect data from all runs, with error handling."""
    runs_data = []

    for run_path in run_paths:
        run_dir = Path(run_path)

        if not run_dir.exists():
            print(f"WARNING: Run directory not found: {run_dir}")
            continue

        metrics = load_metrics(run_dir)
        if metrics is None:
            print(f"WARNING: No metrics.json found in {run_dir}")
            continue

        metadata = load_metadata(run_dir)

        runs_data.append({
            "name": run_dir.name,
            "path": str(run_dir),
            "metrics": metrics,
            "metadata": metadata or {},
        })

    return runs_data


def save_summary_csv(runs_data: list[dict], output_path: Path) -> None:
    """Save summary CSV with one row per run."""
    fieldnames = [
        "name", "model", "dataset", "timestamp",
        "wer", "wer_pct", "rtf_mean", "rtf_p50", "rtf_p95", "num_samples"
    ]

    rows = []
    for run in runs_data:
        row = {
            "name": run["name"],
            "model": run["metadata"].get("model", ""),
            "dataset": run["metadata"].get("dataset", ""),
            "timestamp": run["metadata"].get("timestamp", ""),
            "num_samples": run["metadata"].get("total_samples", ""),
        }

        if "wer" in run["metrics"]:
            row["wer"] = f'{run["metrics"]["wer"]["wer"]:.6f}'
            row["wer_pct"] = f'{run["metrics"]["wer"]["wer"] * 100:.2f}'

        if "rtf" in run["metrics"]:
            row["rtf_mean"] = f'{run["metrics"]["rtf"].get("mean", 0):.6f}'
            row["rtf_p50"] = f'{run["metrics"]["rtf"].get("p50", 0):.6f}'
            row["rtf_p95"] = f'{run["metrics"]["rtf"].get("p95", 0):.6f}'

        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_combined_report(chart_files: list[str], output_dir: Path) -> Path | None:
    """Combine all chart images into a single report.png.

    Args:
        chart_files: List of chart filenames in output_dir.
        output_dir: Directory containing the charts.

    Returns:
        Path to report.png, or None if no charts to combine.
    """
    if not chart_files:
        return None

    # Load all images
    images = []
    for filename in chart_files:
        img_path = output_dir / filename
        if img_path.exists():
            images.append(Image.open(img_path))

    if not images:
        return None

    # Calculate grid layout (prefer 2 columns)
    n_images = len(images)
    n_cols = min(2, n_images)
    n_rows = math.ceil(n_images / n_cols)

    # Get max dimensions for uniform cell sizing
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create combined image
    combined_width = max_width * n_cols
    combined_height = max_height * n_rows
    combined = Image.new("RGB", (combined_width, combined_height), "white")

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        x = col * max_width + (max_width - img.width) // 2
        y = row * max_height + (max_height - img.height) // 2
        combined.paste(img, (x, y))

    # Save combined report
    output_path = output_dir / "report.png"
    combined.save(output_path, dpi=(150, 150))

    # Close images
    for img in images:
        img.close()

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison reports for inference runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to run directories (e.g., outputs/whisper outputs/parakeet)",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Name for report directory (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Base reports directory (default: reports)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Collect run data
    print(f"Loading data from {len(args.runs)} runs...")
    runs_data = collect_run_data(args.runs)

    if not runs_data:
        print("ERROR: No valid runs found. Check run paths.")
        return 1

    print(f"  Loaded {len(runs_data)} runs successfully")

    # Setup output directory
    reports_dir = Path(args.reports_dir)
    report_dir = get_report_dir(reports_dir, args.report_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating report: {report_dir.name}")

    # Generate charts using the metrics registry
    print("  Creating comparison charts...")
    generated_charts = generate_all_charts(runs_data, report_dir)
    for chart in generated_charts:
        print(f"    - {chart}")

    # Create combined report image
    print("  Creating combined report image...")
    combined_report = create_combined_report(generated_charts, report_dir)
    if combined_report:
        print(f"    - {combined_report.name}")

    # Save summary CSV
    print("  Saving summary CSV...")
    save_summary_csv(runs_data, report_dir / "summary.csv")

    # Print summary table
    print(f"\n{'='*60}")
    print("Report Summary")
    print(f"{'='*60}")
    print(f"{'Run':<20} {'WER %':<10} {'RTF Mean':<10}")
    print(f"{'-'*40}")

    for run in runs_data:
        name = run["name"][:19]
        wer = run["metrics"].get("wer", {}).get("wer", None)
        rtf = run["metrics"].get("rtf", {}).get("mean", None)

        wer_str = f"{wer*100:.2f}%" if wer is not None else "N/A"
        rtf_str = f"{rtf:.4f}" if rtf is not None else "N/A"

        print(f"{name:<20} {wer_str:<10} {rtf_str:<10}")

    print(f"\nOutputs saved to: {report_dir}")
    if combined_report:
        print(f"  - {combined_report.name}")
    for chart in generated_charts:
        print(f"  - {chart}")
    print("  - summary.csv")

    return 0


if __name__ == "__main__":
    exit(main())
