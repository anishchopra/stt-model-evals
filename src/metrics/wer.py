"""Word Error Rate (WER) metric implementation."""

from pathlib import Path
from typing import Any

import jiwer
import matplotlib.pyplot as plt

from .base import BaseMetric, MetricResult
from ..text_normalizer import normalize_for_wer


class WERMetric(BaseMetric):
    """Word Error Rate metric for ASR evaluation.

    WER = (Substitutions + Insertions + Deletions) / Reference Words

    Uses jiwer library for computation. Text is normalized before comparison
    (lowercase, punctuation removed) for fair evaluation.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
    ):
        """Initialize WER metric.

        Args:
            lowercase: Convert text to lowercase before comparison.
            remove_punctuation: Remove punctuation before comparison.
        """
        super().__init__(name="wer")
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def compute(
        self,
        predictions: list[dict[str, Any]],
        references: dict[str, dict[str, Any]],
    ) -> MetricResult:
        """Compute WER over the dataset.

        Args:
            predictions: List of dicts with "id" and "prediction" keys.
            references: Dict mapping sample ID to dict with "text" key.

        Returns:
            MetricResult with WER details and per-sample scores.
        """
        # Collect normalized refs and hyps in order
        ref_texts = []
        hyp_texts = []
        sample_ids = []

        for pred in predictions:
            sample_id = pred["id"]
            if sample_id not in references:
                continue

            ref_text = references[sample_id]["text"]
            hyp_text = pred["text"]

            # Normalize for fair comparison
            ref_normalized = normalize_for_wer(
                ref_text,
                lowercase=self.lowercase,
                remove_punctuation=self.remove_punctuation,
            )
            hyp_normalized = normalize_for_wer(
                hyp_text,
                lowercase=self.lowercase,
                remove_punctuation=self.remove_punctuation,
            )

            ref_texts.append(ref_normalized)
            hyp_texts.append(hyp_normalized)
            sample_ids.append(sample_id)

        if not ref_texts:
            return MetricResult(
                name=self.name,
                details={"wer": 0.0, "num_samples": 0},
                per_sample={},
            )

        # Compute aggregate WER using jiwer
        output = jiwer.process_words(ref_texts, hyp_texts)

        # Compute per-sample WER
        per_sample = {}
        for i, sample_id in enumerate(sample_ids):
            sample_output = jiwer.process_words(ref_texts[i], hyp_texts[i])
            per_sample[sample_id] = {
                "wer": sample_output.wer,
                "substitutions": sample_output.substitutions,
                "insertions": sample_output.insertions,
                "deletions": sample_output.deletions,
                "ref_words": len(ref_texts[i].split()),
                "hyp_words": len(hyp_texts[i].split()),
            }

        # Aggregate details
        details = {
            "wer": output.wer,
            "mer": output.mer,  # Match Error Rate
            "wil": output.wil,  # Word Information Lost
            "wip": output.wip,  # Word Information Preserved
            "substitutions": output.substitutions,
            "insertions": output.insertions,
            "deletions": output.deletions,
            "hits": output.hits,
            "total_ref_words": sum(len(r.split()) for r in ref_texts),
            "total_hyp_words": sum(len(h.split()) for h in hyp_texts),
            "num_samples": len(sample_ids),
        }

        return MetricResult(
            name=self.name,
            details=details,
            per_sample=per_sample,
        )

    @staticmethod
    def create_comparison_chart(
        runs_data: list[dict[str, Any]],
        output_path: Path,
    ) -> bool:
        """Generate WER comparison bar chart.

        Args:
            runs_data: List of run data dicts with "name" and "metrics" keys.
            output_path: Path to save the chart image.

        Returns:
            True if chart was created, False if not enough data.
        """
        # Extract WER data from runs
        names = []
        wer_values = []

        for run in runs_data:
            if "wer" in run["metrics"]:
                names.append(run["name"])
                wer_values.append(run["metrics"]["wer"]["wer"] * 100)  # Convert to %

        if not names:
            return False

        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, wer_values, color="steelblue", edgecolor="black")

        # Add value labels on bars
        for bar, val in zip(bars, wer_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Model Run")
        ax.set_ylabel("Word Error Rate (%)")
        ax.set_title("WER Comparison Across Runs")
        ax.set_ylim(0, max(wer_values) * 1.15)  # Add headroom for labels

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return True
