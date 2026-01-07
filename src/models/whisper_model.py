"""Whisper ASR model implementation using faster-whisper."""

import os
import requests
import time
import tempfile
from pathlib import Path

from .base import BaseASRModel, ASRClient, TranscriptionResult


class WhisperClient(ASRClient):
    """HTTP client for Whisper inference server."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Send audio file to server for transcription."""

        start_time = time.perf_counter()

        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            response = requests.post(
                f"{self.base_url}/transcribe",
                files=files,
                timeout=300  # 5 min timeout for long audio
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            raise RuntimeError(f"Transcription failed: {response.text}")

        data = response.json()
        return TranscriptionResult(
            text=data["text"],
            latency_ms=latency_ms,
            audio_duration_s=data.get("audio_duration_s")
        )

    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class WhisperModel(BaseASRModel):
    """Whisper ASR model using faster-whisper backend.

    Uses FastAPI for serving. faster-whisper is up to 4x faster than
    openai-whisper with the same accuracy.

    Available model sizes:
        - tiny, tiny.en
        - base, base.en
        - small, small.en
        - medium, medium.en
        - large-v1, large-v2, large-v3
        - distil-large-v3 (faster, slightly less accurate)
        - turbo (large-v3 optimized)
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        beam_size: int = 5,
    ):
        """Initialize Whisper model.

        Args:
            model_name: Whisper model size (e.g., "base", "small", "large-v3")
            device: "cuda" or "cpu"
            compute_type: "float16", "int8", "int8_float16", or "float32"
            language: Language code for transcription (e.g., "en")
            beam_size: Beam search width (higher = more accurate but slower)
        """
        super().__init__(model_name=model_name, device=device)
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size

    def load_model(self) -> None:
        """Load the Whisper model into memory."""
        from faster_whisper import WhisperModel as FasterWhisperModel

        if self._model is None:
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            self._model = FasterWhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            print(f"Model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        if self._model is None:
            self.load_model()

        segments, info = self._model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            condition_on_previous_text=True,
            vad_filter=True,  # Filter out non-speech
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    def transcribe_with_segments(
        self, audio_path: str
    ) -> tuple[str, list[dict]]:
        """Transcribe with segment-level timestamps.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (full_text, list of segment dicts with start/end/text)
        """
        if self._model is None:
            self.load_model()

        segments, info = self._model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            condition_on_previous_text=True,
            vad_filter=True,
        )

        segment_list = []
        text_parts = []

        for segment in segments:
            text_parts.append(segment.text.strip())
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        return " ".join(text_parts), segment_list

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start FastAPI inference server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        import uvicorn
        from fastapi import FastAPI, UploadFile, File, HTTPException
        from fastapi.responses import JSONResponse

        # Load model before starting server
        self.load_model()

        app = FastAPI(
            title="Whisper ASR Server",
            description=f"Whisper model: {self.model_name}",
        )

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_name}

        @app.get("/info")
        async def info():
            return {
                "model_name": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "language": self.language,
                "beam_size": self.beam_size,
            }

        @app.post("/transcribe")
        async def transcribe_endpoint(file: UploadFile = File(...)):
            # Save uploaded file to temp location
            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name

            try:
                # Get audio duration
                audio_duration = self._get_audio_duration(tmp_path)

                # Transcribe
                start_time = time.perf_counter()
                text = self.transcribe(tmp_path)
                server_latency_ms = (time.perf_counter() - start_time) * 1000

                return JSONResponse({
                    "text": text,
                    "audio_duration_s": audio_duration,
                    "server_latency_ms": server_latency_ms,
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        print(f"Starting Whisper server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    def get_client(self, host: str = "localhost", port: int = 8000) -> ASRClient:
        """Get HTTP client for this model's server.

        Args:
            host: Server host
            port: Server port

        Returns:
            WhisperClient instance
        """
        return WhisperClient(host=host, port=port)
