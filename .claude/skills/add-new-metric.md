---
name: add-new-metric
description: Add a new evaluation metric to the framework. Use when asked to add, implement, or create metrics like latency, semantic similarity, CER (character error rate), LLM-as-judge, or any other ASR evaluation metric.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
---

# Skill: Add New Evaluation Metric

This skill guides you through adding a new evaluation metric to the framework.

## Overview

Adding a new metric requires:
1. Creating a metric class that extends `BaseMetric`
2. Implementing `create_comparison_chart()` for report generation
3. Adding to `METRIC_REGISTRY` in `src/metrics/__init__.py`
4. Adding any required dependencies

The `compute_metrics.py` script automatically picks up all registered metrics, and `generate_report.py` automatically generates comparison charts for each metric.

## Step 1: Create the Metric File

Create `src/metrics/<metric_name>.py`:

```python
"""<MetricName> metric implementation."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .base import BaseMetric, MetricResult


class <MetricName>Metric(BaseMetric):
    """<Brief description of what this metric measures>.

    <Explain the formula or methodology if applicable>
    """

    def __init__(self):
        """Initialize metric."""
        super().__init__(name="<metric_name>")  # lowercase, used as key in output

    def compute(
        self,
        predictions: list[dict[str, Any]],
        references: dict[str, dict[str, Any]],
    ) -> MetricResult:
        """Compute the metric over the dataset.

        Args:
            predictions: List of dicts with keys:
                - "id": Sample identifier (required)
                - "prediction": Model output text (required)
                - "latency_ms": Inference latency (optional)
                - "audio_duration_s": Audio length (optional)
            references: Dict mapping sample ID to dict with keys:
                - "text": Reference transcript (required)
                - "emotion": Emotion label (optional)
                - Additional ground truth fields as needed

        Returns:
            MetricResult with aggregate details and per-sample results.
        """
        per_sample = {}

        # Iterate over predictions
        for pred in predictions:
            sample_id = pred["id"]
            if sample_id not in references:
                continue

            # Get prediction and reference data
            hypothesis = pred["prediction"]
            reference = references[sample_id]["text"]

            # Compute per-sample metric
            sample_score = self._compute_single(hypothesis, reference)
            per_sample[sample_id] = sample_score

        # Compute aggregate statistics
        details = self._aggregate(per_sample)

        return MetricResult(
            name=self.name,
            details=details,
            per_sample=per_sample,
        )

    def _compute_single(self, hypothesis: str, reference: str) -> dict[str, Any]:
        """Compute metric for a single sample.

        Returns:
            Dict with metric values for this sample.
        """
        # Implement your metric calculation here
        score = 0.0  # Replace with actual computation
        return {"score": score}

    def _aggregate(self, per_sample: dict[str, Any]) -> dict[str, Any]:
        """Aggregate per-sample results into summary statistics.

        Returns:
            Dict with aggregate statistics.
        """
        if not per_sample:
            return {"mean": 0.0, "num_samples": 0}

        scores = [s["score"] for s in per_sample.values()]
        return {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "num_samples": len(scores),
        }

    @staticmethod
    def create_comparison_chart(
        runs_data: list[dict[str, Any]],
        output_path: Path,
    ) -> bool:
        """Generate comparison chart for this metric across runs.

        Args:
            runs_data: List of run data dicts, each containing:
                - "name": Run name (str)
                - "metrics": Dict of metric results from metrics.json
            output_path: Path to save the chart image.

        Returns:
            True if chart was created, False if not enough data.
        """
        # Extract metric data from runs
        names = []
        values = []

        for run in runs_data:
            if "<metric_name>" in run["metrics"]:
                names.append(run["name"])
                values.append(run["metrics"]["<metric_name>"]["mean"])

        if not names:
            return False

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, values, color="steelblue", edgecolor="black")

        # Add value labels on bars
        for i, (name, val) in enumerate(zip(names, values)):
            ax.text(i, val + max(values) * 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10)

        ax.set_xlabel("Model Run")
        ax.set_ylabel("<Metric Name>")
        ax.set_title("<Metric Name> Comparison Across Runs")
        ax.set_ylim(0, max(values) * 1.15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return True
```

## Step 2: Register the Metric

Edit `src/metrics/__init__.py`:

```python
from .base import BaseMetric, MetricResult
from .rtf import RTFMetric
from .wer import WERMetric
from .<metric_name> import <MetricName>Metric  # Add import

# Add to registry - this automatically includes it in compute_all_metrics()
METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "wer": WERMetric,
    "rtf": RTFMetric,
    "<metric_name>": <MetricName>Metric,  # Add here
}

__all__ = [
    "BaseMetric",
    "compute_all_metrics",
    "generate_all_charts",
    "METRIC_REGISTRY",
    "MetricResult",
    "RTFMetric",
    "WERMetric",
    "<MetricName>Metric",  # Add to exports
]
```

That's it! The `compute_metrics.py` script uses `compute_all_metrics()` which automatically runs all registered metrics, and `generate_report.py` uses `generate_all_charts()` which automatically creates comparison charts for each metric.

## Step 3: Add Dependencies (if needed)

```bash
uv add <required-package>
```

Common metric dependencies:
- `matplotlib` - Chart generation (already installed)
- `jiwer` - WER, MER, WIL metrics
- `sentence-transformers` - Semantic similarity
- `numpy` - Statistical computations
- `openai` or `anthropic` - LLM-as-judge

## Metric Type Examples

### Text Comparison Metric (like WER)

Uses both `prediction` and `reference["text"]`:

```python
def _compute_single(self, hypothesis: str, reference: str) -> dict[str, Any]:
    # Compare hypothesis to reference
    similarity = compute_similarity(hypothesis, reference)
    return {"similarity": similarity}
```

### Performance Metric (like RTF)

Uses fields from `predictions` only:

```python
def compute(
    self,
    predictions: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> MetricResult:
    rtfs = []
    per_sample = {}

    for pred in predictions:
        latency_ms = pred.get("latency_ms")
        audio_duration_s = pred.get("audio_duration_s")
        if latency_ms and audio_duration_s:
            rtf = (latency_ms / 1000) / audio_duration_s
            rtfs.append(rtf)
            per_sample[pred["id"]] = {"rtf": rtf}

    details = {
        "mean": sum(rtfs) / len(rtfs) if rtfs else 0,
        "num_samples": len(rtfs),
    }

    return MetricResult(name=self.name, details=details, per_sample=per_sample)
```

### LLM-as-Judge Metric

Calls an LLM to evaluate quality:

```python
def _compute_single(self, hypothesis: str, reference: str) -> dict[str, Any]:
    prompt = f"""Rate the transcription accuracy from 1-5.

Reference: {reference}
Transcription: {hypothesis}

Score (1-5):"""

    # Call LLM API
    response = self.client.messages.create(...)
    score = parse_score(response)

    return {"llm_score": score}
```

## MetricResult Structure

The `compute()` method must return a `MetricResult` with:

- `name`: Metric identifier (e.g., "wer", "rtf", "semantic_sim")
- `details`: Dict of aggregate statistics (structure is metric-specific)
- `per_sample`: Dict mapping sample ID to per-sample results

The `details` dict is serialized to `metrics.json`. Include all relevant aggregate statistics. There is no required schema - each metric defines its own structure.

## Checklist

Before considering the metric complete:

- [ ] Class extends `BaseMetric`
- [ ] `__init__` calls `super().__init__(name="<metric_name>")`
- [ ] `compute()` returns `MetricResult` with `details` and `per_sample`
- [ ] `create_comparison_chart()` generates a bar chart for report generation
- [ ] Handles missing samples gracefully (skip if ID not in references)
- [ ] Handles empty input (return sensible defaults)
- [ ] Added to `METRIC_REGISTRY` in `src/metrics/__init__.py`
- [ ] Exported in `__all__` in `src/metrics/__init__.py`

## Testing the New Metric

```python
# Quick test
python -c "
from src.metrics import <MetricName>Metric

predictions = [
    {'id': 's1', 'prediction': 'hello world'},
    {'id': 's2', 'prediction': 'test sentence'},
]
references = {
    's1': {'text': 'hello world'},
    's2': {'text': 'test sentence here'},
}

metric = <MetricName>Metric()
result = metric.compute(predictions, references)
print('Details:', result.details)
print('Per-sample:', result.per_sample)
"
```

```bash
# Full integration test - new metric is automatically included
python -m scripts.compute_metrics --run-name <existing-run>
```
