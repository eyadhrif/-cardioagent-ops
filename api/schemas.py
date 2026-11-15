"""Pydantic v2 models: the JSON "contract" between the gateway, the other services and the client."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class FileInfo(BaseModel):
    """What the gateway itself reads from the uploaded .wav file."""

    filename: str
    size_bytes: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)


class AudioResult(BaseModel):
    """Answer of Service 3 (audio-engine): the ML prediction."""

    label: Literal["normal", "abnormal"]
    probability_abnormal: float = Field(ge=0, le=1)
    model_name: str


class ClinicalReport(BaseModel):
    """Answer of Service 4 (agent-orchestrator): the report written by the LLM agents.

    Anti-hallucination guardrail: an LLM may invent fields or answer "true" as text, so this schema is STRICT.
    """

    # extra="forbid": unknown fields are an error / strict=True: no silent conversion ("true" is not a bool)
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: str = Field(min_length=1)
    risk_level: Literal["low", "medium", "high"]
    recommendations: list[str] = Field(min_length=1)
    sources: list[str] = Field(min_length=1)  # ESC guidelines used by the RAG agent: a report must cite one
    audit_passed: bool  # result of the auditor agent


class Token(BaseModel):
    """One piece of text streamed by the agent-orchestrator (SSE event "token")."""

    text: str


class AnalysisReport(BaseModel):
    """Final structured JSON report returned to the client by POST /analyze_pcg."""

    file: FileInfo
    prompt: str | None = None  # the cleaned user prompt that was sent to the agents (None = no prompt)
    audio: AudioResult
    report: ClinicalReport
    disclaimer: str = "Automatic analysis for education/research only. Not a medical diagnosis."
