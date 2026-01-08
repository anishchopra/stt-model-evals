"""Base class for ASR models with pluggable serving backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time
from torchcodec.decoders import AudioDecoder


@dataclass
class TranscriptionResult:
    """Result from a transcription request."""
    text: str
    latency_ms: float
    audio_duration_s: Optional[float] = None

    @property
    def real_time_factor(self) -> Optional[float]:
        """Ratio of processing time to audio duration. <1 means faster than real-time."""
        if self.audio_duration_s and self.audio_duration_s > 0:
            return (self.latency_ms / 1000) / self.audio_duration_s
        return None


class ASRClient(ABC):
    """Abstract client for communicating with an ASR server."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Send audio to server and get transcription.

        Args:
            audio_path: Path to audio file

        Returns:
            TranscriptionResult with text and timing info
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the server is healthy and ready."""
        pass


class BaseASRModel(ABC):
    """Abstract base class for ASR models.

    Each model implementation chooses its own serving backend
    (FastAPI, vLLM, etc.) while conforming to this interface.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the model.

        Args:
            model_name: Model identifier (e.g., "whisper-small", "parakeet-ctc")
            device: Device to run on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory. Called before serving."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text (direct inference, no server).

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        pass

    @abstractmethod
    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the inference server (blocking).

        Args:
            host: Host to bind to
            port: Port to listen on

        Required endpoints:
            GET /health
                Response: {"status": "healthy", "model": "<model_name>"}

            GET /info
                Response: Model configuration dict (model_name, device, etc.)

            POST /transcribe
                Request: Multipart file upload with "file" field
                Response: {
                    "text": "<transcribed text>",
                    "audio_duration_s": <float or null>,
                    "server_latency_ms": <float>
                }
        """
        pass

    @abstractmethod
    def get_client(self, host: str = "localhost", port: int = 8000) -> ASRClient:
        """Get a client for communicating with this model's server.

        Args:
            host: Server host
            port: Server port

        Returns:
            ASRClient instance configured for this model
        """
        pass

    def transcribe_with_timing(self, audio_path: str) -> TranscriptionResult:
        """Transcribe with latency measurement (direct inference).

        Args:
            audio_path: Path to audio file

        Returns:
            TranscriptionResult with text and timing
        """
        start = time.perf_counter()
        text = self.transcribe(audio_path)
        latency_ms = (time.perf_counter() - start) * 1000

        return TranscriptionResult(
            text=text,
            latency_ms=latency_ms,
            audio_duration_s=self._get_audio_duration(audio_path)
        )

    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio duration in seconds."""
        decoder = AudioDecoder(audio_path)
        metadata = decoder.metadata
        return metadata.duration_seconds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"
