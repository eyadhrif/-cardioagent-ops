"""MOCK of Service 4 (agent-orchestrator). Returns a FAKE report built from the audio prediction.

The real service would run the LangGraph agents (Analyst, RAG Cardiologist, Auditor).
"""

from fastapi import FastAPI

app = FastAPI(title="Mock agent-orchestrator")


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-orchestrator (mock)"}


@app.post("/report")
def report(request: dict):
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
