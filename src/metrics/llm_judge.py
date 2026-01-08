"""LLM-as-Judge metric for semantic equivalence evaluation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm
import matplotlib.pyplot as plt
from openai import OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseMetric, MetricResult

NUM_PARALLEL_CALLS = 100


class JudgmentResult(BaseModel):
    """Structured output for LLM judgment."""

    meaning_score: int
    critical_error: bool
    notes: str


class LLMJudgeMetric(BaseMetric):
    """LLM-as-Judge metric for evaluating semantic equivalence.

    Uses GPT-5-mini to rate how well the hypothesis (model transcript)
    preserves the meaning of the reference (ground truth). Returns a
    meaning score from 1-5 and flags critical errors.
    """

    PROMPT_TEMPLATE = """You are judging semantic equivalence for speech-to-text evaluation.

REFERENCE (ground truth):
<<<{ref}>>>

HYPOTHESIS (model transcript):
<<<{hyp}>>>

Task:
Rate how well the HYPOTHESIS preserves the meaning of the REFERENCE.

Guidelines:
- Ignore punctuation, casing, and minor filler words (uh/um) unless they change meaning.
- Treat added/removed negations, numbers, named entities, or key facts as major issues.
- If the hypothesis is missing a clause that changes intent, score it low.

Output ONLY valid JSON with:
{{
  "meaning_score": 1|2|3|4|5,
  "critical_error": true|false,
  "notes": "brief reason (max 20 words)"
}}

Score meaning_score:
5 = same meaning
4 = mostly same, minor omissions
3 = partial meaning, noticeable omissions/rewrites
2 = mostly different meaning
1 = entirely different / nonsense / contradictory"""

    def __init__(self, model: str = "gpt-5-mini"):
        """Initialize LLM judge metric.

        Args:
            model: OpenAI model to use for judging.
        """
        super().__init__(name="llm_judge")
        self.model = model
        self.client = OpenAI()

    def compute(
        self,
        predictions: list[dict[str, Any]],
        references: dict[str, dict[str, Any]],
    ) -> MetricResult:
        """Compute LLM judge scores over the dataset.

        Args:
            predictions: List of dicts with "id" and "text" keys.
            references: Dict mapping sample ID to dict with "text" key.

        Returns:
            MetricResult with aggregate scores and per-sample judgments.
        """
        per_sample = {}

        # Build list of tasks to process
        tasks = []
        for pred in predictions:
            sample_id = pred["id"]
            if sample_id not in references:
                continue
            tasks.append((sample_id, pred["text"], references[sample_id]["text"]))

        # Process in parallel using thread pool
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_CALLS) as executor:
            futures = {
                executor.submit(self._judge_single, hyp, ref): sample_id
                for sample_id, hyp, ref in tasks
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Judging samples"
            ):
                sample_id = futures[future]
                per_sample[sample_id] = future.result()

        details = self._aggregate(per_sample)

        return MetricResult(
            name=self.name,
            details=details,
            per_sample=per_sample,
        )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(10),
    )
    def _judge_single(self, hypothesis: str, reference: str) -> dict[str, Any]:
        """Get LLM judgment for a single sample.

        Args:
            hypothesis: Model transcript.
            reference: Ground truth transcript.

        Returns:
            Dict with meaning_score, critical_error, and notes.
        """
        prompt = self.PROMPT_TEMPLATE.format(ref=reference, hyp=hypothesis)

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=JudgmentResult,
        )

        result = response.choices[0].message.parsed

        return {
            "meaning_score": result.meaning_score,
            "critical_error": result.critical_error,
            "notes": result.notes,
        }

    def _aggregate(self, per_sample: dict[str, Any]) -> dict[str, Any]:
        """Aggregate per-sample results into summary statistics.

        Returns:
            Dict with aggregate statistics.
        """
        if not per_sample:
            return {
                "mean_score": 0.0,
                "critical_error_rate": 0.0,
                "num_samples": 0,
            }

        scores = [s["meaning_score"] for s in per_sample.values()]
        critical_errors = sum(1 for s in per_sample.values() if s["critical_error"])

        return {
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "critical_error_count": critical_errors,
            "critical_error_rate": critical_errors / len(scores),
            "score_distribution": {
                str(i): sum(1 for s in scores if s == i) for i in range(1, 6)
            },
            "num_samples": len(scores),
        }

    @staticmethod
    def create_comparison_chart(
        runs_data: list[dict[str, Any]],
        output_dir: Path,
    ) -> list[Path]:
        """Generate LLM judge comparison charts.

        Args:
            runs_data: List of run data dicts with "name" and "metrics" keys.
            output_dir: Directory to save the chart images.

        Returns:
            List of paths to generated chart images.
        """
        names = []
        scores = []
        critical_errors = []

        for run in runs_data:
            if "llm_judge" in run["metrics"]:
                names.append(run["name"])
                scores.append(run["metrics"]["llm_judge"]["mean_score"])
                critical_errors.append(run["metrics"]["llm_judge"]["critical_error_count"])

        if not names:
            return []

        generated = []

        # Chart 1: Mean meaning score
        score_path = output_dir / "llm_judge_score_comparison.png"
        _fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, scores, color="mediumseagreen", edgecolor="black")

        for bar, val in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Model Run")
        ax.set_ylabel("Mean Meaning Score (1-5)")
        ax.set_title("LLM Judge Score Comparison Across Runs")
        ax.set_ylim(0, 5.5)

        plt.tight_layout()
        plt.savefig(score_path, dpi=150, bbox_inches="tight")
        plt.close()
        generated.append(score_path)

        # Chart 2: Critical error count
        error_path = output_dir / "llm_judge_critical_errors_comparison.png"
        _fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, critical_errors, color="indianred", edgecolor="black")

        for bar, val in zip(bars, critical_errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(critical_errors) * 0.02,
                str(val),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Model Run")
        ax.set_ylabel("Critical Error Count")
        ax.set_title("LLM Judge Critical Errors Comparison Across Runs")
        if max(critical_errors) > 0:
            ax.set_ylim(0, max(critical_errors) * 1.15)

        plt.tight_layout()
        plt.savefig(error_path, dpi=150, bbox_inches="tight")
        plt.close()
        generated.append(error_path)

        return generated
