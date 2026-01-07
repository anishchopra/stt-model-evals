"""Real-Time Factor (RTF) metric implementation."""

from typing import Any

from .base import BaseMetric, MetricResult


class RTFMetric(BaseMetric):
    """Real-Time Factor metric for ASR evaluation.

    RTF = processing_time / audio_duration

    RTF < 1 means faster than real-time (good)
    RTF = 1 means real-time
    RTF > 1 means slower than real-time
    """

    def __init__(self):
        """Initialize RTF metric."""
        super().__init__(name="rtf")

    def compute(
        self,
        predictions: list[dict[str, Any]],
        references: dict[str, dict[str, Any]],
    ) -> MetricResult:
        """Compute RTF statistics over the dataset.

        Args:
            predictions: List of dicts with "latency_ms" and "audio_duration_s"
            references: Not used for RTF, but required by interface

        Returns:
            MetricResult with RTF statistics and per-sample results.
        """
        rtfs = []
        per_sample = {}

        for pred in predictions:
            sample_id = pred["id"]
            latency_ms = pred.get("latency_ms")
            audio_duration_s = pred.get("audio_duration_s")

            # RTF requires both latency and audio duration
            if latency_ms is None or audio_duration_s is None or audio_duration_s <= 0:
                continue

            latency_s = latency_ms / 1000
            rtf = latency_s / audio_duration_s
            rtfs.append(rtf)

            per_sample[sample_id] = {
                "rtf": rtf,
            }

        # Compute aggregate statistics
        details = self._compute_stats(rtfs)

        return MetricResult(
            name=self.name,
            details=details,
            per_sample=per_sample,
        )

    def _compute_stats(self, rtfs: list[float]) -> dict[str, Any]:
        """Compute aggregate RTF statistics."""
        if not rtfs:
            return {"num_samples": 0}

        rtfs_sorted = sorted(rtfs)
        n = len(rtfs_sorted)

        return {
            "num_samples": n,
            "mean": sum(rtfs) / n,
            "min": rtfs_sorted[0],
            "max": rtfs_sorted[-1],
            "p50": self._percentile(rtfs_sorted, 0.5),
            "p90": self._percentile(rtfs_sorted, 0.9),
            "p95": self._percentile(rtfs_sorted, 0.95),
            "p99": self._percentile(rtfs_sorted, 0.99),
        }

    def _percentile(self, sorted_values: list[float], p: float) -> float:
        """Compute percentile from sorted values."""
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        idx = int(n * p)
        idx = min(idx, n - 1)
        return sorted_values[idx]
