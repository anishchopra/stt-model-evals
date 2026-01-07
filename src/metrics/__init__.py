"""Metrics for ASR evaluation."""

from typing import Any

from .base import BaseMetric, MetricResult
from .rtf import RTFMetric
from .wer import WERMetric

# Registry of all available metrics
# Add new metrics here to include them in compute_all_metrics()
METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "wer": WERMetric,
    "rtf": RTFMetric,
}


def compute_all_metrics(
    predictions: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> dict[str, MetricResult]:
    """Compute all registered metrics.

    Args:
        predictions: List of prediction dicts with "id", "prediction", etc.
        references: Dict mapping sample ID to reference dict with "text", etc.

    Returns:
        Dict mapping metric name to MetricResult.
    """
    results = {}
    for name, metric_class in METRIC_REGISTRY.items():
        metric = metric_class()
        results[name] = metric.compute(predictions, references)
    return results


__all__ = [
    "BaseMetric",
    "compute_all_metrics",
    "METRIC_REGISTRY",
    "MetricResult",
    "RTFMetric",
    "WERMetric",
]
