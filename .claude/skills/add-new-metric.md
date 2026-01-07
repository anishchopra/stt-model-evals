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
2. Exporting the metric in `__init__.py`
3. Integrating into `compute_metrics.py`
4. Adding any required dependencies

## Step 1: Create the Metric File

Create `src/metrics/<metric_name>.py`:

```python
"""<MetricName> metric implementation."""

from typing import Any

from .base import BaseMetric, MetricResult


class <MetricName>Metric(BaseMetric):
    """<Brief description of what this metric measures>.

    <Explain the formula or methodology if applicable>
    """

    def __init__(
        self,
        # Add metric-specific configuration parameters
        param1: str = "default",
    ):
        """Initialize metric.

        Args:
            param1: Description of parameter.
        """
        super().__init__(name="<metric_name>")  # lowercase, used as key in output
        self.param1 = param1

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
```

## Step 2: Export the Metric

Edit `src/metrics/__init__.py`:

```python
from .base import BaseMetric, MetricResult
from .wer import WERMetric
from .<metric_name> import <MetricName>Metric  # Add import

__all__ = [
    "BaseMetric",
    "MetricResult",
    "WERMetric",
    "<MetricName>Metric",  # Add to exports
]
```

## Step 3: Integrate into compute_metrics.py

Edit `scripts/compute_metrics.py` to use the new metric:

```python
from src.metrics import WERMetric, <MetricName>Metric

# In main(), after WER computation:
<metric_name>_metric = <MetricName>Metric()
<metric_name>_result = <metric_name>_metric.compute(predictions, references)
metrics_results["<metric_name>"] = <metric_name>_result.summary()
print(f"  <MetricName>: {<metric_name>_result.details['mean']:.4f}")

# Add to per_sample_results:
per_sample_results["<metric_name>"] = <metric_name>_result.per_sample
```

## Step 4: Add Dependencies (if needed)

```bash
uv add <required-package>
```

Common metric dependencies:
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

### Performance Metric (like Latency)

Uses fields from `predictions` only:

```python
def compute(
    self,
    predictions: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> MetricResult:
    latencies = []
    per_sample = {}

    for pred in predictions:
        latency = pred.get("latency_ms")
        if latency is not None:
            latencies.append(latency)
            per_sample[pred["id"]] = {"latency_ms": latency}

    # Compute percentiles
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    details = {
        "mean_ms": sum(latencies) / n if n else 0,
        "p50_ms": latencies_sorted[n // 2] if n else 0,
        "p95_ms": latencies_sorted[int(n * 0.95)] if n else 0,
        "num_samples": n,
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

- `name`: Metric identifier (e.g., "wer", "latency", "semantic_sim")
- `details`: Dict of aggregate statistics (structure is metric-specific)
- `per_sample`: Dict mapping sample ID to per-sample results

The `details` dict is serialized to `metrics.json`. Include all relevant aggregate statistics. There is no required schema - each metric defines its own structure.

## Checklist

Before considering the metric complete:

- [ ] Class extends `BaseMetric`
- [ ] `__init__` calls `super().__init__(name="<metric_name>")`
- [ ] `compute()` returns `MetricResult` with `details` and `per_sample`
- [ ] Handles missing samples gracefully (skip if ID not in references)
- [ ] Handles empty input (return sensible defaults)
- [ ] Exported in `src/metrics/__init__.py`
- [ ] Integrated into `scripts/compute_metrics.py`
- [ ] Dependencies added via `uv add`

## Testing the New Metric

```python
# Quick test
uv run python -c "
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
# Full integration test
uv run python -m scripts.compute_metrics --run-name <existing-run>
```
