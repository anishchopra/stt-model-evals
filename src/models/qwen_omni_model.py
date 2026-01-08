"""Qwen3 Omni ASR model implementation using Hugging Face transformers.

Qwen3-Omni is a multimodal model capable of processing audio for transcription.
This implementation uses the transformers library with flash attention support.
"""

import os
import requests
import time
import tempfile
from pathlib import Path

from .base import BaseASRModel, ASRClient, TranscriptionResult

# Check for availability at import time
_QWEN_OMNI_AVAILABLE = False
try:
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    _QWEN_OMNI_AVAILABLE = True
except ImportError:
    Qwen3OmniMoeForConditionalGeneration = None
    Qwen3OmniMoeProcessor = None


class QwenOmniClient(ASRClient):
    """HTTP client for Qwen Omni inference server."""

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


class QwenOmniModel(BaseASRModel):
    """Qwen3 Omni ASR model using Hugging Face transformers backend.

    Uses FastAPI for serving. Qwen3-Omni is a multimodal MoE model that can
    process audio input for transcription tasks.

    Available model variants:
        - Qwen/Qwen3-Omni-30B-A3B-Instruct (default)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
    ):
        """Initialize Qwen Omni model.

        Args:
            model_name: Qwen Omni model name from Hugging Face
                       (e.g., "Qwen/Qwen3-Omni-30B-A3B-Instruct")
            device: "cuda" or "cpu"
            torch_dtype: Data type for model weights ("bfloat16", "float16", "float32")
            use_flash_attention: Whether to use flash attention 2
        """
        super().__init__(model_name=model_name, device=device)
        self.torch_dtype = torch_dtype
        self.use_flash_attention = use_flash_attention
        self._processor = None

    def load_model(self) -> None:
        """Load the Qwen3 Omni model into memory."""
        if not _QWEN_OMNI_AVAILABLE:
            raise ImportError(
                "Qwen3 Omni transformers classes are required but not available. "
                "Install with: pip install transformers>=4.51.0\n"
                "Also ensure you have: pip install accelerate qwen-omni-utils"
            )

        if self._model is None:
            # Load processor
            

            # Load model with flash attention
            self.use_flash_attention = False
            attn_implementation = "flash_attention_2" if self.use_flash_attention else "sdpa"

            self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
                attn_implementation=attn_implementation,
            )

            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_name)

            if self.device == "cpu":
                self._model = self._model.to("cpu")

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

        from qwen_omni_utils import process_mm_info  # type: ignore[import-not-found]

        # Build conversation for transcription
        conversation = [
            {
                "role": "system",
                "content": "You are a transcription assistant. Transcribe the audio exactly as spoken, without adding any commentary or explanations.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            },
        ]

        # Process the conversation through the template
        text = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Process multimedia info (audio)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

        # Prepare inputs
        inputs = self._processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )

        # Move inputs to model device and dtype
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        # Generate transcription
        import torch
        with torch.inference_mode():
            text_ids, audio = self._model.generate(
                **inputs,
                max_new_tokens=512,
                use_audio_in_video=True,
                return_audio=False,
            )

        # Decode the output, skipping the input tokens
        transcription = self._processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return transcription.strip()

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
            title="Qwen3 Omni ASR Server",
            description=f"Qwen3 Omni model: {self.model_name}",
        )

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_name}

        @app.get("/info")
        async def info():
            return {
                "model_name": self.model_name,
                "device": self.device,
                "torch_dtype": self.torch_dtype,
                "use_flash_attention": self.use_flash_attention,
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

        print(f"Starting Qwen3 Omni server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    def get_client(self, host: str = "localhost", port: int = 8000) -> ASRClient:
        """Get HTTP client for this model's server.

        Args:
            host: Server host
            port: Server port

        Returns:
            QwenOmniClient instance
        """
        return QwenOmniClient(host=host, port=port)
