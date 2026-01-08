"""Parakeet ASR model implementation using NVIDIA NeMo.

Note: Requires nemo_toolkit[asr] to be installed. NeMo may have compatibility
issues with Python 3.12+. If you encounter installation issues, try using
Python 3.10 or 3.11, or install NeMo from source following NVIDIA's instructions:
https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html
"""

import os
import requests
import time
import tempfile
from pathlib import Path

from .base import BaseASRModel, ASRClient, TranscriptionResult

# Check for NeMo availability at import time
_NEMO_AVAILABLE = False
try:
    import nemo.collections.asr as nemo_asr
    _NEMO_AVAILABLE = True
except ImportError:
    nemo_asr = None


class ParakeetClient(ASRClient):
    """HTTP client for Parakeet inference server."""

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


class ParakeetModel(BaseASRModel):
    """Parakeet ASR model using NVIDIA NeMo backend.

    Uses FastAPI for serving. Parakeet models are NVIDIA's state-of-the-art
    ASR models trained on large-scale English speech data.

    Available model sizes:
        - parakeet-ctc-0.6b (CTC decoder, 600M params)
        - parakeet-ctc-1.1b (CTC decoder, 1.1B params)
        - parakeet-rnnt-0.6b (RNN-T decoder, 600M params)
        - parakeet-rnnt-1.1b (RNN-T decoder, 1.1B params)
        - parakeet-tdt-1.1b (TDT decoder with timestamps, 1.1B params)
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-1.1b",
        device: str = "cuda",
    ):
        """Initialize Parakeet model.

        Args:
            model_name: Parakeet model name from NGC/HuggingFace
                       (e.g., "nvidia/parakeet-ctc-1.1b")
            device: "cuda" or "cpu"
        """
        super().__init__(model_name=model_name, device=device)

    def load_model(self) -> None:
        """Load the Parakeet model into memory."""
        if not _NEMO_AVAILABLE:
            raise ImportError(
                "NeMo toolkit is required for Parakeet models but is not installed. "
                "Install it with: pip install nemo_toolkit[asr]\n"
                "Note: NeMo may have compatibility issues with Python 3.12+. "
                "See https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html"
            )

        import torch

        if self._model is None:
            print(f"Loading Parakeet model '{self.model_name}' on {self.device}...")
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            # Move to specified device
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
            # Set to evaluation mode
            self._model.eval()
            print("Model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        if self._model is None:
            self.load_model()

        # NeMo ASR models have a transcribe method that takes a list of paths
        transcriptions = self._model.transcribe([audio_path])

        return transcriptions[0].text

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
            title="Parakeet ASR Server",
            description=f"Parakeet model: {self.model_name}",
        )

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_name}

        @app.get("/info")
        async def info():
            return {
                "model_name": self.model_name,
                "device": self.device,
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
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        print(f"Starting Parakeet server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    def get_client(self, host: str = "localhost", port: int = 8000) -> ASRClient:
        """Get HTTP client for this model's server.

        Args:
            host: Server host
            port: Server port

        Returns:
            ParakeetClient instance
        """
        return ParakeetClient(host=host, port=port)
