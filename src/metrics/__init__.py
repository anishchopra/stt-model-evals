"""Metrics for ASR evaluation."""

from pathlib import Path
from typing import Any

from .base import BaseMetric, MetricResult
from .llm_judge import LLMJudgeMetric
from .rtf import RTFMetric
from .wer import WERMetric

# Registry of all available metrics
# Add new metrics here to include them in compute_all_metrics()
METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "wer": WERMetric,
    "rtf": RTFMetric,
    "llm_judge": LLMJudgeMetric,
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


def generate_all_charts(
    runs_data: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Generate comparison charts for all registered metrics.

    Args:
        runs_data: List of run data dicts, each containing:
            - "name": Run name (str)
            - "metrics": Dict of metric results from metrics.json
        output_dir: Directory to save chart images.

    Returns:
        List of generated chart filenames.
    """
    generated = []
    for name, metric_class in METRIC_REGISTRY.items():
        paths = metric_class.create_comparison_chart(runs_data, output_dir)
        generated.extend(p.name for p in paths)

    return generated


__all__ = [
    "BaseMetric",
    "compute_all_metrics",
    "generate_all_charts",
    "LLMJudgeMetric",
    "METRIC_REGISTRY",
    "MetricResult",
    "RTFMetric",
    "WERMetric",
]
