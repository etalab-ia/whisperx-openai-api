"""Tests for POST /v1/audio/transcriptions."""

ENDPOINT = "/v1/audio/transcriptions"
AUDIO = b"\x00" * 512


def post(client, model=None, language=None, response_format=None):
    data = {}
    if model:
        data["model"] = model
    if language:
        data["language"] = language
    if response_format:
        data["response_format"] = response_format
    return client.post(ENDPOINT, files={"file": ("test.wav", AUDIO, "audio/wav")}, data=data)


def test_transcription(client):
    body = post(client).json()
    assert body["text"] == "Hello world."
    assert body["segments"] is None


def test_diarized(client):
    body = post(client, response_format="diarized_json").json()
    assert body["segments"][0]["text"] == "Hello world."
    assert body["segments"][0]["speaker"] == "SPEAKER_00"


def test_text(client):
    response = post(client, response_format="text")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Hello world."


def test_unsupported_format_returns_400(client):
    assert post(client, response_format="srt").status_code == 400


def test_wrong_model_returns_404(client):
    assert post(client, model="tiny").status_code == 404


def test_english(client):
    assert post(client, language="en").status_code == 200


def test_french(client):
    assert post(client, language="fr").status_code == 200


def test_missing_file_returns_422(client):
    assert client.post(ENDPOINT).status_code == 422
