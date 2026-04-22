# Whisper OpenAI API

FastAPI-based wrapper around WhisperX, providing an openAI compatible API for transcription and speaker diarization.

## Getting Started

### Install uv

Install instructions [in this link](https://docs.astral.sh/uv/getting-started/installation/).

### API-only Development

Inference libraries (`whisperx`, `pytorch`, etc.) are heavy and may not run on all devices. We provide a `dev` dependency group to allow running API tests locally and IDE autocompletion. To install:

```bash
uv sync --group dev
```

### Full Inference Development

To develop with a fully functional transcription pipeline:

```bash
uv sync --group dev --group inference
```

Run the server locally:

```bash
export PORT=8010
export RELOAD=true
export LOGGING_CONFIG=logging-config.yaml
python app/main.py
```

## Testing

Tests mock actual inference and can be run locally:

```bash
cd app
python -m pytest tests/ -v
```

### Integration tests

Check the [documentation to run integration tests](docs/testing_with_gpu.md) on GPU.

## Environment Variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| API_KEY | API key for API access | Required |
| HF_TOKEN | Hugging Face token (required for diarization) | Required |
| TRANSCRIBE_MODEL | WhisperX model to load | `large-v2` |
| BATCH_SIZE | Transcription batch size | `16` |
| DIARIZE_MODEL | Pyannote diarization model | `pyannote/speaker-diarization-community-1` |
| PRELOADED_ALIGN_MODEL_LANGUAGES | Languages to pre-load alignment models for | `["en", "fr", "nl", "de"]` |
| RETURN_CHAR_ALIGNMENTS | Return character-level alignments (diarization only) | `false` |
| INTERPOLATE_METHOD | WhisperX interpolation method (diarization only) | `nearest` |
| FILL_NEAREST | Fill nearest gaps in speaker assignment (diarization only) | `false` |
| TIMEOUT_KEEP_ALIVE | Keep-alive timeout (seconds) | `60` |
| PORT | Server port | `8000` |
| WORKERS | Number of uvicorn workers (each loads its own model in VRAM) | `1` |
| RELOAD | Enable auto-reload | `false` |
| ROOT_PATH | API root path | `None` |
| LOGGING_CONFIG | Path to logging config file | `None` |
| DEBUG | Enable debug logging | `false` |

