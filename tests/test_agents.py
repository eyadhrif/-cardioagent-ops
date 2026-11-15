"""Conformity tests of the agents' output (brief: test_agents.py).

1. The ClinicalReport schema is strict: it rejects what an LLM could hallucinate.
2. The agent-orchestrator (today the mock, later the real LangGraph service) always answers in that schema,
   on /report and on the /report/stream SSE stream.
"""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.schemas import ClinicalReport, Token
from mocks.orchestrator import app

orchestrator = TestClient(app)

VALID = {
    "summary": "Systolic murmur detected.",
    "risk_level": "high",
    "recommendations": ["Echocardiography"],
    "sources": ["ESC 2021 valvular heart disease guidelines"],
    "audit_passed": True,
}


def request_body(label: str, probability: float, prompt: str | None = None) -> dict:
    audio = {"label": label, "probability_abnormal": probability, "model_name": "test"}
    return {"filename": "heart.wav", "audio": audio, "prompt": prompt}


# ---------- 1. the schema rejects hallucinated or malformed output ----------

def test_valid_report_is_accepted():
    assert ClinicalReport.model_validate(VALID).risk_level == "high"


@pytest.mark.parametrize(
    "change",
    [
        {"diagnosis": "aortic stenosis"},     # invented field
        {"risk_level": "critical"},           # value outside the allowed list
        {"audit_passed": "true"},             # text instead of a real boolean
        {"summary": ""},                      # empty text
        {"recommendations": []},              # no recommendation
        {"sources": []},                      # no ESC source: not grounded by the RAG
        {"sources": "ESC 2021"},              # a string instead of a list
    ],
)
def test_hallucinated_or_malformed_report_is_rejected(change):
    with pytest.raises(ValidationError):
        ClinicalReport.model_validate({**VALID, **change})


@pytest.mark.parametrize("missing", list(VALID))
def test_every_field_is_required(missing):
    report = {key: value for key, value in VALID.items() if key != missing}
    with pytest.raises(ValidationError):
        ClinicalReport.model_validate(report)


# ---------- 2. the orchestrator always answers in the schema ----------

@pytest.mark.parametrize("label", ["normal", "abnormal"])
@pytest.mark.parametrize("probability", [0.0, 0.49, 0.5, 0.69, 0.7, 1.0])
@pytest.mark.parametrize("prompt", [None, "Patient has chest pain"])
def test_report_conforms_to_schema(label, probability, prompt):
    response = orchestrator.post("/report", json=request_body(label, probability, prompt))
    assert response.status_code == 200
    ClinicalReport.model_validate(response.json())


@pytest.mark.parametrize(
    "probability, risk", [(0.1, "low"), (0.49, "low"), (0.5, "medium"), (0.69, "medium"), (0.7, "high")]
)
def test_risk_level_follows_probability(probability, risk):
    report = orchestrator.post("/report", json=request_body("abnormal", probability)).json()
    assert report["risk_level"] == risk


def test_prompt_is_used_as_context():
    report = orchestrator.post("/report", json=request_body("normal", 0.1, "shortness of breath")).json()
    assert "shortness of breath" in report["summary"]


def test_stream_conforms_and_matches_the_report():
    body = request_body("abnormal", 0.83, "chest pain")
    response = orchestrator.post("/report/stream", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")

    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))

    *tokens, (last_name, last_data) = events
    assert all(name == "token" for name, _ in tokens) and last_name == "report"
    for _, data in tokens:
        Token.model_validate(data)
    report = ClinicalReport.model_validate(last_data)

    # The streamed text is exactly the summary, and the streamed report equals the /report answer
    assert "".join(data["text"] for _, data in tokens).strip() == report.summary
    assert last_data == orchestrator.post("/report", json=body).json()


def test_health():
    assert orchestrator.get("/health").json()["status"] == "ok"
