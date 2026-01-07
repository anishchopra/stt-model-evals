# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STT Model Evals is a Speech-to-Text model evaluation framework for benchmarking ASR models (Whisper, Parakeet, etc.) on audio datasets. It uses a server-client architecture where models serve via frameworks such as vLLM and FastAPI and inference runs against a dataset to produce predictions, which are then evaluated with metrics like WER.

## Key Commands

```bash
# Always use uv for running Python
uv run python -m scripts.serve_model --model whisper --port 8000
uv run python -m scripts.run_inference --model whisper --port 8000 --limit 10
uv run python -m scripts.compute_metrics --run-name <run-name>

# Add dependencies
uv add <package-name>
```

## Architecture

### Model System (`src/models/`)
- `BaseASRModel`: Abstract base with `load_model()`, `transcribe()`, `serve()`, `get_client()`
- `MODEL_REGISTRY` and `MODEL_DEFAULTS` in `__init__.py` - central registration for all models
- Each model implements its own FastAPI server and HTTP client

### Metrics System (`src/metrics/`)
- `BaseMetric`: Abstract base with `compute(predictions, references) -> MetricResult`
- `MetricResult`: Contains `details` (aggregate stats) and `per_sample` (per-sample scores)
- References passed as `dict[str, dict[str, Any]]` with `"text"` key for flexibility

### Data Flow
1. `serve_model.py` starts HTTP server for a model
2. `run_inference.py` sends audio files to server, saves predictions to `outputs/<run-name>/`
3. `compute_metrics.py` loads predictions + references, computes metrics, saves to same directory

### Output Structure
```
outputs/<run-name>/
├── predictions.jsonl      # {id, prediction, latency_ms, audio_duration_s}
├── metadata.json          # run config, dataset info, timestamps
├── metrics.json           # aggregate metrics
└── metrics_per_sample.json
```

### Text Normalization (`src/text_normalizer.py`)
Reference transcripts contain `**emphasis**` markers and `[paralinguistic]` tags that must be stripped before WER calculation. Use `normalize_for_wer()` for fair comparison.

## Conventions

- Run scripts as modules: `uv run python -m scripts.<name>` (not `python scripts/<name>.py`)
- Never edit `pyproject.toml` directly - use `uv add`
- Run names auto-increment if directory exists (whisper, whisper0, whisper1, ...)
- `--run-name` is optional for inference (defaults to model name), required for metrics
