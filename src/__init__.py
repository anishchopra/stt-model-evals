"""STT Model Evaluation package."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"
load_dotenv(_env_file)

from .data_loader import EvalDataset, AudioSample, load_eval_dataset
from .text_normalizer import (
    clean_reference_transcript,
    normalize_for_wer,
    prepare_for_evaluation,
)

__all__ = [
    "EvalDataset",
    "AudioSample",
    "load_eval_dataset",
    "clean_reference_transcript",
    "normalize_for_wer",
    "prepare_for_evaluation",
]
