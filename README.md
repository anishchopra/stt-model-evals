# STT Model Evaluation Framework

A reproducible benchmarking pipeline for evaluating Speech-to-Text (ASR) models on English audio data.

## Overview

This framework provides:
- Server-client architecture for model inference (FastAPI/vLLM backends)
- Pluggable model support (Whisper, Parakeet, etc.)
- Extensible metrics system (WER, latency, semantic similarity, etc.)
- Structured output with per-sample and aggregate results

## Requirements

- Python 3.12+
- CUDA-capable GPU (tested on A100)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd stt-model-evals

# Install dependencies with pip
pip install -r requirements.txt
```

## Project Structure

```
stt-model-evals/
├── data/                      # Default directory with labeled audio data
│   ├── eval_10h/              # Audio files ({id}.wav)
│   └── eval_10h.csv           # Manifest with reference transcripts
│
├── src/
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── text_normalizer.py     # Text cleaning for WER computation
│   ├── models/
│   │   ├── base.py            # BaseASRModel, ASRClient abstract classes
│   │   ├── whisper_model.py   # Whisper implementation (faster-whisper)
│   │   └── __init__.py        # Model registry
│   └── metrics/
│       ├── base.py            # BaseMetric abstract class
│       ├── wer.py             # Word Error Rate metric
│       └── __init__.py        # Metric exports
│
├── scripts/
│   ├── serve_model.py         # Start inference server
│   ├── run_inference.py       # Run inference on dataset
│   ├── compute_metrics.py     # Calculate metrics on predictions
│   └── generate_report.py     # Generate comparison reports
│
├── outputs/                   # Evaluation results (auto-created)
│   └── <run-name>/
│       ├── metadata.json
│       ├── predictions.jsonl
│       ├── metrics.json
│       └── metrics_per_sample.json
│
└── reports/                   # Comparison reports (auto-created)
    └── <report-name>/
        ├── wer_comparison.png
        ├── rtf_comparison.png
        └── summary.csv
```

## Data Format

The evaluation dataset consists of:

1. **Audio files**: `data/eval_10h/{id}.wav`
2. **Manifest CSV**: `data/eval_10h.csv` with columns:
   - `id` - Sample identifier (matches audio filename)
   - `emotion` - Emotion label
   - `qa_edited_transcript` - Reference transcript

Reference transcripts may contain:
- `**word**` - Emphasis markers (stripped during preprocessing)
- `[tag]` - Paralinguistic tags like `[inhale]`, `[chuckle]` (stripped during preprocessing)

## Running the Full Pipeline

### Step 1: Start the Model Server

Start an inference server for your chosen model:

```bash
# Start Whisper server (default: base model, port 8000)
python -m scripts.serve_model --model whisper

# With custom options
python -m scripts.serve_model \
    --model whisper \
    --model-name large-v3 \
    --port 8001 \
    --compute-type float16 \
    --beam-size 5
```

Server options:
- `--model` - Model type from registry (required)
- `--model-name` - Specific model variant (default: from MODEL_DEFAULTS)
- `--port` - Server port (default: 8000)
- `--device` - Device to use (default: cuda)
- `--compute-type` - Precision for Whisper (default: float16)
- `--beam-size` - Beam size for decoding (default: 5)
- `--language` - Language code (default: en)

### Step 2: Run Inference

In a separate terminal, run inference on the dataset:

```bash
# Basic usage (creates outputs/whisper/)
python -m scripts.run_inference --model whisper

# With custom run name and options
python -m scripts.run_inference \
    --model whisper \
    --run-name whisper-large-v3 \
    --port 8000 \
    --limit 100  # Process only first 100 samples
```

Inference options:
- `--model` - Model type (required, for client selection)
- `--run-name` - Output directory name (default: model name, auto-increments if exists)
- `--port` - Server port (default: 8000)
- `--host` - Server host (default: localhost)
- `--limit` - Limit number of samples (optional, for testing)

Output files:
- `outputs/<run-name>/metadata.json` - Run configuration
- `outputs/<run-name>/predictions.jsonl` - Per-sample predictions (streaming)

### Step 3: Compute Metrics

After inference completes, compute metrics:

```bash
python -m scripts.compute_metrics --run-name whisper
```

This generates:
- `outputs/<run-name>/metrics.json` - Aggregate metrics
- `outputs/<run-name>/metrics_per_sample.json` - Per-sample metrics

### Step 4: Generate Comparison Reports (Optional)

Compare metrics across multiple runs:

```bash
# Compare two or more runs
python -m scripts.generate_report --runs outputs/whisper outputs/parakeet

# With custom report name
python -m scripts.generate_report --runs outputs/whisper outputs/parakeet --report-name whisper-vs-parakeet
```

Report options:
- `--runs` - Paths to run directories (required, space-separated)
- `--report-name` - Report directory name (default: auto-generated timestamp)
- `--reports-dir` - Base reports directory (default: reports)

This generates:
- `reports/<report-name>/wer_comparison.png` - WER bar chart
- `reports/<report-name>/rtf_comparison.png` - RTF bar chart
- `reports/<report-name>/summary.csv` - Tabular summary of all runs

## Available Models

| Model | Registry Key | Backend | Notes |
|-------|-------------|---------|-------|
| Whisper | `whisper` | FastAPI + faster-whisper | Supports all Whisper sizes |
| Qwen3 Omni | `qwen-omni` | FastAPI + transformers | Qwen3-Omni multimodal MoE model |

To add a new model, see `.claude/skills/add-new-model.md`.

## Available Metrics

| Metric | Description |
|--------|-------------|
| WER | Word Error Rate (using jiwer) |
| RTF | Real-Time Factor with percentiles (p50, p90, p95, p99) |
| LLM Judge | Semantic equivalence scoring (1-5) using GPT-5-mini |

**RTF (Real-Time Factor)** = processing_time / audio_duration. RTF < 1 means faster than real-time.

**LLM Judge** uses GPT-5-mini to rate how well transcriptions preserve meaning (1=nonsense, 5=same meaning). Requires `OPENAI_API_KEY` environment variable.

To add a new metric, see `.claude/skills/add-new-metric.md`.

## Example: Complete Evaluation Run

```bash
# Terminal 1: Start server
python -m scripts.serve_model --model whisper --model-name base

# Terminal 2: Run evaluation
python -m scripts.run_inference --model whisper --run-name whisper-base
python -m scripts.compute_metrics --run-name whisper-base

# View results
cat outputs/whisper-base/metrics.json

# Compare multiple runs (after running evaluations for different models)
python -m scripts.generate_report --runs outputs/whisper-base outputs/parakeet-base
```

## Output Format

### predictions.jsonl

Each line contains:
```json
{
  "id": "sample_id",
  "prediction": "transcribed text",
  "latency_ms": 123.45,
  "audio_duration_s": 5.67
}
```

### metrics.json

```json
{
  "wer": {
    "name": "wer",
    "wer": 0.032,
    "mer": 0.031,
    "wil": 0.046,
    "wip": 0.953,
    "substitutions": 9,
    "insertions": 3,
    "deletions": 7,
    "hits": 579,
    "total_ref_words": 595,
    "total_hyp_words": 591,
    "num_samples": 10
  },
  "rtf": {
    "name": "rtf",
    "num_samples": 10,
    "mean": 0.018,
    "min": 0.015,
    "max": 0.022,
    "p50": 0.017,
    "p90": 0.020,
    "p95": 0.021,
    "p99": 0.022
  }
}
```

### metrics_per_sample.json

```json
{
  "wer": {
    "sample_id_1": {
      "wer": 0.073,
      "substitutions": 2,
      "insertions": 1,
      "deletions": 0,
      "ref_words": 41,
      "hyp_words": 42
    }
  },
  "rtf": {
    "sample_id_1": {
      "rtf": 0.017,
      "latency_ms": 445.2,
      "audio_duration_s": 26.2
    }
  }
}
```

## Extending the Framework

### Adding a New Model

1. Create `src/models/<model_name>_model.py`
2. Implement `<ModelName>Client(ASRClient)` and `<ModelName>Model(BaseASRModel)`
3. Register in `src/models/__init__.py`

See `.claude/skills/add-new-model.md` for detailed instructions.

### Adding a New Metric

1. Create `src/metrics/<metric_name>.py`
2. Implement `<MetricName>Metric(BaseMetric)`
3. Export in `src/metrics/__init__.py`
4. Integrate into `scripts/compute_metrics.py`

See `.claude/skills/add-new-metric.md` for detailed instructions.