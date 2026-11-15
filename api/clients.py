"""Integration code: how the gateway talks to audio-engine and agent-orchestrator with httpx."""

import json
import os
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel, ValidationError

from api.schemas import AudioResult, ClinicalReport, Token

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


def validate_answer(model: type[BaseModel], data, service: str):
    """Pydantic guardrail: every answer of another service must match our schema exactly."""
    try:
        return model.model_validate(data)
    except ValidationError:
        raise ServiceError(502, f"{service} returned an unexpected format")


async def call_audio_engine(filename: str, content: bytes) -> AudioResult:
    data = await _post(
        "audio-engine",
        f"{AUDIO_ENGINE_URL}/predict",
        files={"file": (filename, content, "audio/wav")},
    )
    return validate_answer(AudioResult, data, "audio-engine")


def _orchestrator_payload(filename: str, audio: AudioResult, prompt: str | None) -> dict:
    return {"filename": filename, "audio": audio.model_dump(), "prompt": prompt}


async def call_orchestrator(filename: str, audio: AudioResult, prompt: str | None = None) -> ClinicalReport:
    data = await _post(
        "agent-orchestrator", f"{ORCHESTRATOR_URL}/report", json=_orchestrator_payload(filename, audio, prompt)
    )
    return validate_answer(ClinicalReport, data, "agent-orchestrator")


async def stream_orchestrator(
    filename: str, audio: AudioResult, prompt: str | None = None
) -> AsyncIterator[Token | ClinicalReport]:
    """Read the orchestrator's Server-Sent Events and yield each Token, then the final ClinicalReport.

    SSE format: blocks separated by an empty line, each block has "event: <name>" and "data: <json>" lines.
    """
    service = "agent-orchestrator"
    got_report = False
    try:
        async with make_client() as client:
            async with client.stream(
                "POST", f"{ORCHESTRATOR_URL}/report/stream", json=_orchestrator_payload(filename, audio, prompt)
            ) as response:
                if response.status_code != 200:
                    raise ServiceError(502, f"{service} returned an error (HTTP {response.status_code})")

                event, data = "message", ""
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event = line.removeprefix("event:").strip()
                    elif line.startswith("data:"):
                        data += line.removeprefix("data:").strip()
                    elif line == "" and data:  # empty line = end of one event
                        try:
                            payload = json.loads(data)
                        except ValueError:
                            raise ServiceError(502, f"{service} returned invalid JSON")
                        if event == "token":
                            yield validate_answer(Token, payload, service)
                        elif event == "report":
                            got_report = True
                            yield validate_answer(ClinicalReport, payload, service)
                        event, data = "message", ""
    except httpx.TimeoutException:
        raise ServiceError(504, f"{service} did not answer in time")
    except httpx.RequestError:
        raise ServiceError(503, f"{service} is unavailable")

    if not got_report:
        raise ServiceError(502, f"{service} ended the stream without a report")
