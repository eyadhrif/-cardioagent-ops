"""Integration tests of the api-gateway (Service 5) with pytest + HTTPX.

The other services are replaced by a fake httpx transport, so no network or Docker is needed.
"""

import io
import wave

import httpx
import pytest
from fastapi.testclient import TestClient  # built on HTTPX

from api import clients, main
from api.main import app

client = TestClient(app)

GOOD_AUDIO = {"label": "abnormal", "probability_abnormal": 0.83, "model_name": "test-model"}
GOOD_REPORT = {
    "summary": "Murmur detected",
    "risk_level": "high",
    "recommendations": ["See a cardiologist"],
    "sources": ["ESC 2021"],
    "audit_passed": True,
}


def make_wav(seconds: float = 1.0, sample_rate: int = 2000) -> bytes:
    """Create a small silent .wav file in memory (16-bit mono)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buffer.getvalue()


def upload(filename: str = "heart.wav", content: bytes | None = None):
    if content is None:
        content = make_wav()
    return client.post("/analyze_pcg", files={"file": (filename, content, "audio/wav")})


def fake_services(monkeypatch, handler):
    """Make the gateway's httpx client use `handler` instead of the real network."""
    monkeypatch.setattr(
        clients, "make_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def happy_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/predict":
        return httpx.Response(200, json=GOOD_AUDIO)
    return httpx.Response(200, json=GOOD_REPORT)


# ---------- basic endpoints ----------

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "api-gateway"}


def test_home_page_is_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "analyze_pcg" in response.text


# ---------- file checks ----------

def test_rejects_wrong_extension():
    response = upload(filename="heart.mp3")
    assert response.status_code == 400
    assert "wav" in response.json()["detail"]


def test_rejects_empty_file():
    response = upload(content=b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_rejects_fake_wav():
    response = upload(content=b"this is not audio")
    assert response.status_code == 400
    assert "not a valid WAV" in response.json()["detail"]


def test_rejects_wav_without_audio():
    response = upload(content=make_wav(seconds=0))
    assert response.status_code == 400
    assert "no audio" in response.json()["detail"]


def test_rejects_too_large_file(monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_SIZE", 10)
    response = upload()
    assert response.status_code == 413


def test_missing_file_field():
    response = client.post("/analyze_pcg")
    assert response.status_code == 422  # FastAPI validation error


# ---------- full flow ----------

def test_analyze_pcg_success(monkeypatch):
    fake_services(monkeypatch, happy_handler)
    response = upload(content=make_wav(seconds=2.5))

    assert response.status_code == 200
    body = response.json()
    assert body["file"]["filename"] == "heart.wav"
    assert body["file"]["sample_rate"] == 2000
    assert body["file"]["duration_seconds"] == 2.5
    assert body["audio"] == GOOD_AUDIO
    assert body["report"] == GOOD_REPORT
    assert "Not a medical diagnosis" in body["disclaimer"]


def test_gateway_sends_audio_prediction_to_orchestrator(monkeypatch):
    received = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/report":
            received["body"] = request.content
        return happy_handler(request)

    fake_services(monkeypatch, handler)
    upload()
    assert b"probability_abnormal" in received["body"]


# ---------- errors from the other services ----------

def test_audio_engine_down(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    fake_services(monkeypatch, handler)
    response = upload()
    assert response.status_code == 503
    assert "audio-engine is unavailable" in response.json()["detail"]


def test_audio_engine_timeout(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    fake_services(monkeypatch, handler)
    response = upload()
    assert response.status_code == 504


def test_audio_engine_http_error(monkeypatch):
    fake_services(monkeypatch, lambda request: httpx.Response(500))
    response = upload()
    assert response.status_code == 502


def test_audio_engine_invalid_json(monkeypatch):
    fake_services(monkeypatch, lambda request: httpx.Response(200, content=b"not json"))
    response = upload()
    assert response.status_code == 502
    assert "invalid JSON" in response.json()["detail"]


@pytest.mark.parametrize(
    "bad_audio",
    [
        {"label": "maybe", "probability_abnormal": 0.5, "model_name": "m"},  # unknown label
        {"label": "normal", "probability_abnormal": 1.7, "model_name": "m"},  # probability > 1
        {"label": "normal"},  # missing fields
    ],
)
def test_audio_engine_wrong_format(monkeypatch, bad_audio):
    fake_services(monkeypatch, lambda request: httpx.Response(200, json=bad_audio))
    response = upload()
    assert response.status_code == 502
    assert "unexpected format" in response.json()["detail"]


def test_orchestrator_wrong_format(monkeypatch):
    def handler(request):
        if request.url.path == "/predict":
            return httpx.Response(200, json=GOOD_AUDIO)
        return httpx.Response(200, json={"summary": "no risk level"})

    fake_services(monkeypatch, handler)
    response = upload()
    assert response.status_code == 502
    assert "agent-orchestrator" in response.json()["detail"]
