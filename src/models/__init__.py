"""ASR model implementations."""

from .base import BaseASRModel, ASRClient, TranscriptionResult
from .whisper_model import WhisperModel, WhisperClient
from .parakeet_model import ParakeetModel, ParakeetClient
from .qwen_omni_model import QwenOmniModel, QwenOmniClient


# Registry of available models
MODEL_REGISTRY: dict[str, type[BaseASRModel]] = {
    "whisper": WhisperModel,
    "parakeet": ParakeetModel,
    "qwen-omni": QwenOmniModel,
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
    "parakeet": {
        "model_name": "nvidia/parakeet-ctc-1.1b",
        "device": "cuda",
    },
    "qwen-omni": {
        "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "device": "cuda",
        "torch_dtype": "bfloat16",
        "use_flash_attention": True,
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
    "ParakeetModel",
    "ParakeetClient",
    "QwenOmniModel",
    "QwenOmniClient",
    "MODEL_REGISTRY",
    "MODEL_DEFAULTS",
    "get_model_class",
    "get_model_defaults",
]
