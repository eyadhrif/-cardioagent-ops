"""Integration code: how the gateway talks to audio-engine and agent-orchestrator with httpx."""

import os

import httpx
from pydantic import ValidationError

from api.schemas import AudioResult, ClinicalReport

# Inside Docker Compose these are set to http://audio-engine:8001 etc. (service names on the docker-network).
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://localhost:8001")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8002")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds


class ServiceError(Exception):
    """Raised when another service fails. The gateway turns it into an HTTP error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def make_client() -> httpx.AsyncClient:
    # Separate function so the tests can replace it with a fake client.
    return httpx.AsyncClient(timeout=REQUEST_TIMEOUT)


async def _post(service: str, url: str, **kwargs) -> dict:
    """Send a POST and translate every network problem into a clear ServiceError."""
    try:
        async with make_client() as client:
            response = await client.post(url, **kwargs)
    except httpx.TimeoutException:
        raise ServiceError(504, f"{service} did not answer in time")
    except httpx.RequestError:
        raise ServiceError(503, f"{service} is unavailable")

    if response.status_code != 200:
        raise ServiceError(502, f"{service} returned an error (HTTP {response.status_code})")
    try:
        return response.json()
    except ValueError:
        raise ServiceError(502, f"{service} returned invalid JSON")


async def call_audio_engine(filename: str, content: bytes) -> AudioResult:
    data = await _post(
        "audio-engine",
        f"{AUDIO_ENGINE_URL}/predict",
        files={"file": (filename, content, "audio/wav")},
    )
    try:
        return AudioResult.model_validate(data)  # Pydantic guardrail on the answer
    except ValidationError:
        raise ServiceError(502, "audio-engine returned an unexpected format")


async def call_orchestrator(filename: str, audio: AudioResult, prompt: str | None = None) -> ClinicalReport:
    data = await _post(
        "agent-orchestrator",
        f"{ORCHESTRATOR_URL}/report",
        json={"filename": filename, "audio": audio.model_dump(), "prompt": prompt},
    )
    try:
        return ClinicalReport.model_validate(data)
    except ValidationError:
        raise ServiceError(502, "agent-orchestrator returned an unexpected format")
