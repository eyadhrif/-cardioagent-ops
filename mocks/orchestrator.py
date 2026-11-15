"""MOCK of Service 4 (agent-orchestrator). Returns a FAKE report built from the audio prediction.

The real service would run the LangGraph agents (Analyst, RAG Cardiologist, Auditor) and stream the
LLM tokens. This mock keeps the same contract: POST /report (JSON) and POST /report/stream (SSE).
"""

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(title="Mock agent-orchestrator")

TOKEN_DELAY_SECONDS = 0.05  # pause between two streamed words, to look like an LLM writing


def build_report(request: dict) -> dict:
    audio = request["audio"]
    probability = audio["probability_abnormal"]

    if probability >= 0.7:
        risk = "high"
    elif probability >= 0.5:
        risk = "medium"
    else:
        risk = "low"

    if audio["label"] == "abnormal":
        summary = f"Abnormal heart sound detected (probability {probability:.0%})."
        recommendations = ["Consult a cardiologist", "Perform an echocardiography"]
    else:
        summary = f"No abnormality detected (abnormal probability {probability:.0%})."
        recommendations = ["No immediate action", "Routine follow-up"]

    # The real agents would use the prompt as clinical context; the mock only shows it was received
    prompt = request.get("prompt")
    if prompt:
        summary += f" Context from the user: {prompt}."

    return {
        "summary": summary + " [MOCK REPORT]",
        "risk_level": risk,
        "recommendations": recommendations,
        "sources": ["ESC Guidelines for the management of valvular heart disease (mock reference)"],
        "audit_passed": True,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-orchestrator (mock)"}


@app.post("/report")
def report(request: dict):
    return build_report(request)


@app.post("/report/stream")
async def report_stream(request: dict):
    """Server-Sent Events: the summary word by word ("token"), then the full report ("report")."""
    full_report = build_report(request)

    async def events():
        for word in full_report["summary"].split(" "):
            yield f"event: token\ndata: {json.dumps({'text': word + ' '})}\n\n"
            await asyncio.sleep(TOKEN_DELAY_SECONDS)
        yield f"event: report\ndata: {json.dumps(full_report)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
