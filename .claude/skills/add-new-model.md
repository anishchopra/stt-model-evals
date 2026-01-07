---
name: add-new-model
description: Add support for a new ASR model type to the evaluation framework. Use when asked to add, implement, or integrate a new speech-to-text model like Parakeet, Kimi-Audio, Qwen-Omni, or any other ASR model.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
---

# Skill: Add New ASR Model Type

This skill guides you through adding support for a new ASR model to the evaluation framework.

## Overview

Adding a new model requires:
1. Creating a model file with two classes (Model + Client)
2. Registering the model in the registry
3. Adding required dependencies

## Step 1: Create the Model File

Create `src/models/<model_name>_model.py` with two classes:

### 1.1 Client Class (extends `ASRClient`)

The client communicates with the model's HTTP server.

```python
"""<ModelName> ASR model implementation."""

import os
import requests
import time
from pathlib import Path

from .base import BaseASRModel, ASRClient, TranscriptionResult


class <ModelName>Client(ASRClient):
    """HTTP client for <ModelName> inference server."""

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
```

**Required client methods:**
- `transcribe(audio_path: str) -> TranscriptionResult` - Must return `TranscriptionResult` with `text`, `latency_ms`, and optionally `audio_duration_s`
- `health_check() -> bool` - Must return `True` if server is ready

### 1.2 Model Class (extends `BaseASRModel`)

The model class handles loading, inference, and serving.

```python
class <ModelName>Model(BaseASRModel):
    """<ModelName> ASR model using <backend> backend."""

    def __init__(
        self,
        model_name: str = "default-variant",
        device: str = "cuda",
        # Add model-specific parameters here
    ):
        super().__init__(model_name=model_name, device=device)
        # Store model-specific config
        self.custom_param = custom_param

    def load_model(self) -> None:
        """Load the model into memory."""
        if self._model is None:
            print(f"Loading {self.model_name} on {self.device}...")
            # Import and instantiate the actual model
            from some_library import SomeModel
            self._model = SomeModel(self.model_name, device=self.device)
            print("Model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        if self._model is None:
            self.load_model()

        # Call the underlying model's transcription method
        result = self._model.transcribe(audio_path)

        # Return plain text string
        return result.text

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start inference server."""
        import uvicorn
        from fastapi import FastAPI, UploadFile, File, HTTPException
        from fastapi.responses import JSONResponse
        import tempfile

        self.load_model()

        app = FastAPI(
            title="<ModelName> ASR Server",
            description=f"Model: {self.model_name}",
        )

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_name}

        @app.get("/info")
        async def info():
            return {
                "model_name": self.model_name,
                "device": self.device,
                # Include model-specific config
            }

        @app.post("/transcribe")
        async def transcribe_endpoint(file: UploadFile = File(...)):
            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

                try:
                    audio_duration = self._get_audio_duration(tmp_path)
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

        print(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    def get_client(self, host: str = "localhost", port: int = 8000) -> ASRClient:
        """Get HTTP client for this model's server."""
        return <ModelName>Client(host=host, port=port)
```

**Required model methods:**
- `__init__`: Must call `super().__init__(model_name=model_name, device=device)`
- `load_model()`: Load model into `self._model`, must be idempotent (check `if self._model is None`)
- `transcribe(audio_path: str) -> str`: Return plain text transcription
- `serve(host, port)`: Start FastAPI server with `/health`, `/info`, `/transcribe` endpoints
- `get_client(host, port) -> ASRClient`: Return corresponding client instance

**Server endpoint requirements:**
- `GET /health`: Return `{"status": "healthy", "model": "<model_name>"}`
- `GET /info`: Return model configuration dict
- `POST /transcribe`: Accept file upload, return `{"text": "...", "audio_duration_s": ..., "server_latency_ms": ...}`

## Step 2: Register the Model

Edit `src/models/__init__.py`:

```python
from .base import BaseASRModel, ASRClient, TranscriptionResult
from .whisper_model import WhisperModel, WhisperClient
from .<model_name>_model import <ModelName>Model, <ModelName>Client  # Add import

MODEL_REGISTRY: dict[str, type[BaseASRModel]] = {
    "whisper": WhisperModel,
    "<model_name>": <ModelName>Model,  # Add to registry
}

MODEL_DEFAULTS: dict[str, dict] = {
    "whisper": {...},
    "<model_name>": {  # Add default config
        "model_name": "default-variant",
        "device": "cuda",
        # Model-specific defaults
    },
}

__all__ = [
    # ... existing exports
    "<ModelName>Model",
    "<ModelName>Client",
]
```

**Registry key naming:**
- Use lowercase, hyphen-separated names (e.g., `"whisper"`, `"parakeet"`, `"kimi-audio"`)
- This key is used in CLI: `--model <key>`

## Step 3: Add Dependencies

```bash
uv add <required-package>
```

Common ASR dependencies:
- `faster-whisper` - Whisper models
- `nemo_toolkit[asr]` - NVIDIA NeMo models (Parakeet)
- `transformers` - Hugging Face models

## Checklist

Before considering the model complete:

- [ ] Client class implements `transcribe()` returning `TranscriptionResult`
- [ ] Client class implements `health_check()` returning `bool`
- [ ] Model class calls `super().__init__(model_name=model_name, device=device)`
- [ ] Model class `load_model()` is idempotent (checks `self._model is None`)
- [ ] Model class `transcribe()` returns plain `str`
- [ ] Model class `serve()` exposes `/health`, `/info`, `/transcribe` endpoints
- [ ] Model class `get_client()` returns the corresponding client
- [ ] Model registered in `MODEL_REGISTRY`
- [ ] Default config added to `MODEL_DEFAULTS`
- [ ] Classes exported in `__all__`
- [ ] Dependencies added via `uv add`

## Testing the New Model

```bash
# Terminal 1: Start server
uv run python -m scripts.serve_model --model <model_name> --port 8000

# Terminal 2: Test health
curl http://localhost:8000/health

# Terminal 2: Test transcription
curl -X POST http://localhost:8000/transcribe -F "file=@data/eval_10h/sample.wav"

# Terminal 2: Run inference
uv run python -m scripts.run_inference --model <model_name> --port 8000 --limit 5
```

## Notes on vLLM Integration

For LLM-based audio models (Qwen-Omni, Kimi-Audio), the `serve()` method can use vLLM instead of FastAPI:

```python
def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start vLLM inference server."""
    from vllm import LLM, SamplingParams
    # vLLM-specific serving logic
```

The client remains the same HTTP-based approach, just targeting vLLM's API format instead.
