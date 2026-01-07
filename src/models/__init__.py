"""ASR model implementations."""

from .base import BaseASRModel, ASRClient, TranscriptionResult
from .whisper_model import WhisperModel, WhisperClient


# Registry of available models
MODEL_REGISTRY: dict[str, type[BaseASRModel]] = {
    "whisper": WhisperModel,
    # "parakeet": ParakeetModel,  # TODO: add later
}

# Default configurations for each model
MODEL_DEFAULTS: dict[str, dict] = {
    "whisper": {
        "model_name": "base",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
        "beam_size": 5,
    },
}


def get_model_class(model_name: str) -> type[BaseASRModel]:
    """Get model class by name."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_REGISTRY[model_name]


def get_model_defaults(model_name: str) -> dict:
    """Get default configuration for a model."""
    return MODEL_DEFAULTS.get(model_name, {}).copy()


__all__ = [
    "BaseASRModel",
    "ASRClient",
    "TranscriptionResult",
    "WhisperModel",
    "WhisperClient",
    "MODEL_REGISTRY",
    "MODEL_DEFAULTS",
    "get_model_class",
    "get_model_defaults",
]
