import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock libs before import
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
_mock_torch.float32 = "float32"

_mock_whisperx = MagicMock()
_mock_whisperx.utils.LANGUAGES = {"en": "english", "fr": "french", "cz": "czech"}
_mock_whisperx.alignment.DEFAULT_ALIGN_MODELS_HF = {"en": "...", "fr": "..."}
_mock_whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH = {}

for name, mock in {
    "numpy": MagicMock(),
    "torch": _mock_torch,
    "whisperx": _mock_whisperx,
    "whisperx.utils": _mock_whisperx.utils,
    "whisperx.alignment": _mock_whisperx.alignment,
    "whisperx.asr": _mock_whisperx.asr,
    "whisperx.diarize": _mock_whisperx.diarize,
}.items():
    sys.modules.setdefault(name, mock)

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("HF_TOKEN", "test-token")

# App imports after mocking)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from endpoints import audio  # noqa: E402
from utils.config import Settings, get_settings  # noqa: E402
from utils.security import check_api_key  # noqa: E402

FAKE_TRANSCRIPTION = {
    "language": "en",
    "segments": [{"start": 0.0, "end": 1.5, "text": "Hello world.", "speaker": "SPEAKER_00"}],
}


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(audio.router, prefix="/v1")
    app.dependency_overrides[get_settings] = lambda: Settings(
        api_key="test-key", hf_token="test-token", transcribe_model="large-v3-turbo"
    )
    app.dependency_overrides[check_api_key] = lambda: "test-key"

    with (
        patch.object(audio.whisperx, "load_audio", return_value=MagicMock()),
        patch.object(audio, "transcribe", return_value=FAKE_TRANSCRIPTION),
        TestClient(app) as c,
    ):
        yield c
