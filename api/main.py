"""Service 5: api-gateway. Single entry point of CardioAgent-Ops."""

import io
import json
import wave
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from api.clients import ServiceError, call_audio_engine, call_orchestrator, stream_orchestrator
from api.schemas import AnalysisReport, FileInfo, HealthResponse, Token

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB (longest PCG in our dataset is ~2 minutes)
MAX_PROMPT_LENGTH = 500  # characters: enough for a short clinical context
# Prompt-injection guardrail: phrases that try to take control of the LLM agents behind the gateway
FORBIDDEN_PHRASES = (
    "ignore previous instructions",
    "ignore all previous",
    "ignore the above",
    "disregard previous",
    "system prompt",
    "you are now",
)
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="CardioAgent-Ops API Gateway", version="1.0.0")


def validate_wav(filename: str | None, content: bytes) -> FileInfo:
    """Check the upload is a real, non-empty .wav file and read its header."""
    if not filename or not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The file is empty")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="The file is too large (max 20 MB)")

    # Same idea as notebooks/01_eda_data_wrangling.ipynb: open it with the standard "wave" module.
    try:
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            n_frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except (wave.Error, EOFError):
        raise HTTPException(status_code=400, detail="The file is not a valid WAV audio file")
    if n_frames == 0 or sample_rate == 0:
        raise HTTPException(status_code=400, detail="The WAV file contains no audio")

    return FileInfo(
        filename=filename,
        size_bytes=len(content),
        sample_rate=sample_rate,
        duration_seconds=round(n_frames / sample_rate, 3),
    )


def validate_prompt(prompt: str | None) -> str | None:
    """Check the optional user text. Returns the cleaned prompt, or None when there is no prompt."""
    if prompt is None:
        return None
    # Collapse spaces, tabs and new lines: "ignore\n   previous" becomes "ignore previous"
    prompt = " ".join(prompt.split())
    if not prompt:
        return None  # an empty text box means "no prompt", not an error
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"The prompt is too long (max {MAX_PROMPT_LENGTH} characters)"
        )
    lowered = prompt.lower()
    if any(phrase in lowered for phrase in FORBIDDEN_PHRASES):
        raise HTTPException(status_code=400, detail="The prompt contains forbidden instructions")
    return prompt


@app.get("/", include_in_schema=False)
def home():
    """Small HTML upload page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", service="api-gateway")


async def check_inputs(file: UploadFile, prompt: str | None) -> tuple[FileInfo, bytes, str | None]:
    """Guardrails shared by both endpoints: the .wav file and the optional prompt."""
    content = await file.read()
    file_info = validate_wav(file.filename, content)
    return file_info, content, validate_prompt(prompt)


FILE_FIELD = File(..., description="Heart sound recording (.wav)")
PROMPT_FIELD = Form(None, description="Optional clinical context, e.g. 'patient has chest pain'")


@app.post("/analyze_pcg", response_model=AnalysisReport)
async def analyze_pcg(file: UploadFile = FILE_FIELD, prompt: str | None = PROMPT_FIELD):
    # 1. Receive and check the inputs (guardrails): the .wav file and the optional prompt
    file_info, content, clean_prompt = await check_inputs(file, prompt)

    try:
        # 2-3. Send the audio to audio-engine -> prediction (label + probability)
        audio = await call_audio_engine(file_info.filename, content)
        # 4-7. Send the prediction (+ prompt) to agent-orchestrator -> clinical report (agents + RAG + LLM)
        report = await call_orchestrator(file_info.filename, audio, clean_prompt)
    except ServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)

    # 8. Return the structured JSON report (validated by Pydantic)
    return AnalysisReport(file=file_info, prompt=clean_prompt, audio=audio, report=report)


# ---------- Streaming version (Server-Sent Events) ----------

def sse(event: str, data: dict) -> str:
    """One Server-Sent Event: an event name, a JSON data line, then an empty line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def analysis_events(file_info: FileInfo, content: bytes, prompt: str | None) -> AsyncIterator[str]:
    """The same flow as /analyze_pcg, but each result is sent to the client as soon as it is ready."""
    yield sse("file", file_info.model_dump())
    try:
        audio = await call_audio_engine(file_info.filename, content)
        yield sse("audio", audio.model_dump())

        async for item in stream_orchestrator(file_info.filename, audio, prompt):
            if isinstance(item, Token):
                yield sse("token", item.model_dump())  # the report text, piece by piece
            else:
                report = AnalysisReport(file=file_info, prompt=prompt, audio=audio, report=item)
                yield sse("report", report.model_dump())
    except ServiceError as error:
        # The HTTP status (200) is already sent once streaming starts, so errors become an "error" event
        yield sse("error", {"status_code": error.status_code, "detail": error.message})


@app.post("/analyze_pcg/stream")
async def analyze_pcg_stream(file: UploadFile = FILE_FIELD, prompt: str | None = PROMPT_FIELD):
    """Same inputs as /analyze_pcg. Answer: text/event-stream with events file, audio, token..., report (or error)."""
    # Input errors are still normal HTTP errors (400/413/422): they are checked BEFORE streaming starts
    file_info, content, clean_prompt = await check_inputs(file, prompt)
    return StreamingResponse(
        analysis_events(file_info, content, clean_prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # no caching/buffering by proxies
    )
