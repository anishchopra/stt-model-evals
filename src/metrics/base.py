"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Result from computing a metric.

    Attributes:
        name: Metric name (e.g., "wer", "latency")
        details: Aggregate statistics (structure defined by each metric)
        per_sample: Per-sample results keyed by sample ID
    """
    name: str
    details: dict[str, Any] = field(default_factory=dict)
    per_sample: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Get summary dict for serialization."""
        return {
            "name": self.name,
            **self.details,
        }


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    Metrics compute scores comparing model predictions against references,
    or analyzing prediction properties (like latency).
    """

    def __init__(self, name: str):
        """Initialize metric.

        Args:
            name: Metric identifier (e.g., "wer", "latency")
        """
        self.name = name

    @abstractmethod
    def compute(
        self,
        predictions: list[dict[str, Any]],
        references: dict[str, dict[str, Any]],
    ) -> MetricResult:
        """Compute the metric over a dataset.

        Args:
            predictions: List of prediction dicts with at least "id" and "text" keys.
                        May also contain "latency_ms", "audio_duration_s", etc.
            references: Dict mapping sample ID to reference dict containing at least
                       a "text" key. May include other ground truth data.

        Returns:
            MetricResult with aggregate details and per-sample results.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
