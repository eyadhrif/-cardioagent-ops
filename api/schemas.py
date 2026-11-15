"""Pydantic v2 models: the JSON "contract" between the gateway, the other services and the client."""

from typing import Literal

from pydantic import BaseModel, Field


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
    """Answer of Service 4 (agent-orchestrator): the report written by the agents."""

    summary: str
    risk_level: Literal["low", "medium", "high"]
    recommendations: list[str]
    sources: list[str]  # ESC guidelines used by the RAG cardiologist agent
    audit_passed: bool  # result of the auditor agent


class AnalysisReport(BaseModel):
    """Final structured JSON report returned to the client by POST /analyze_pcg."""

    file: FileInfo
    audio: AudioResult
    report: ClinicalReport
    disclaimer: str = "Automatic analysis for education/research only. Not a medical diagnosis."
