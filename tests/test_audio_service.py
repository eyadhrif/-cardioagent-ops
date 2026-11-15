"""Tests of the real audio-engine: ONNX export (export_onnx.py) and the HTTP service (service.py).

A tiny model is trained on synthetic features, so no real dataset and no MLflow server are needed.
"""

import io
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audio_engine import export_onnx, service
from audio_engine.features import SAMPLE_RATE, feature_names


def fake_features(n_files: int = 24) -> pd.DataFrame:
    """Abnormal files have shifted features, so the model is easy to learn."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n_files):
        label = int(i % 3 == 0)
        for seg in range(2):
            values = rng.normal(loc=2.0 * label, size=30)
            rows.append({"file": f"f{i}.wav", "segment": seg, "label": label,
                         **dict(zip(feature_names(), values))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tiny_model():
    df = fake_features()
    X, y = df[feature_names()], df["label"]
    model = export_onnx.train_final_model(X, y)
    return model, export_onnx.to_onnx_bytes(model, X.shape[1]), X


@pytest.fixture
def client(tiny_model, tmp_path, monkeypatch):
    """The service, using the tiny ONNX model."""
    path = tmp_path / "model.onnx"
    path.write_bytes(tiny_model[1])
    monkeypatch.setattr(service, "MODEL_PATH", path)
    service.get_session.cache_clear()  # forget a model loaded by another test
    yield TestClient(service.app)
    service.get_session.cache_clear()


def wav_bytes(seconds: float, sample_rate: int = 4000) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    buffer = io.BytesIO()
    sf.write(buffer, 0.5 * np.sin(2 * np.pi * 50 * t), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------- export_onnx.py ----------

def test_final_model_has_probabilities(tiny_model):
    model, _, X = tiny_model
    assert model.named_steps["svm"].probability is True
    assert model.predict_proba(X).shape == (len(X), 2)


def test_onnx_gives_the_same_answers_as_sklearn(tiny_model):
    model, onnx_bytes, X = tiny_model
    check = export_onnx.compare_with_sklearn(onnx_bytes, model, X)
    assert check["label_agreement"] == 1.0
    assert check["max_probability_diff"] < 1e-4  # float32 vs float64


def test_export_main_writes_model_and_logs_to_mlflow(monkeypatch, tmp_path):
    fake_mlflow = MagicMock()
    monkeypatch.setattr(export_onnx, "mlflow", fake_mlflow)
    monkeypatch.setattr(export_onnx, "build_dataset", lambda rebuild=False: fake_features())
    monkeypatch.setattr(export_onnx, "MODEL_PATH", tmp_path / "cardio_model.onnx")

    check = export_onnx.main([])

    assert (tmp_path / "cardio_model.onnx").stat().st_size > 0
    assert check["label_agreement"] == 1.0
    fake_mlflow.start_run.assert_called_once_with(run_name="export-onnx-rbf")
    fake_mlflow.log_artifact.assert_called_once_with(str(tmp_path / "cardio_model.onnx"))


# ---------- service.py ----------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "audio-engine", "model_loaded": True}


def test_predict_real_wav(client):
    response = client.post("/predict", files={"file": ("heart.wav", wav_bytes(7.5), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"normal", "abnormal"}
    assert 0 <= body["probability_abnormal"] <= 1
    assert body["model_name"] == service.MODEL_NAME
    assert body["n_segments"] == 2  # 7.5 s = one 5 s segment + one padded 2.5 s segment


def test_predict_answer_matches_gateway_contract(client):
    from api.schemas import AudioResult  # the gateway's schema: the two services must agree

    response = client.post("/predict", files={"file": ("heart.wav", wav_bytes(3), "audio/wav")})
    AudioResult.model_validate(response.json())


def test_predict_rejects_non_audio(client):
    response = client.post("/predict", files={"file": ("x.wav", b"not audio at all", "audio/wav")})
    assert response.status_code == 400


def test_predict_rejects_audio_without_samples(client):
    response = client.post("/predict", files={"file": ("x.wav", wav_bytes(0), "audio/wav")})
    assert response.status_code == 400


class FakeSession:
    """Stands in for ONNX Runtime: returns fixed labels, to test the vote rule alone."""

    def __init__(self, labels, p_abnormal):
        self.labels, self.p = np.array(labels), np.array(p_abnormal)

    def get_inputs(self):
        return [MagicMock(name="input")]

    def run(self, outputs, feeds):
        return self.labels, np.column_stack([1 - self.p, self.p])


@pytest.mark.parametrize(
    "labels, expected",
    [([1, 1, 0], "abnormal"), ([0, 0, 1], "normal"), ([1, 0], "abnormal")],  # a tie counts as abnormal
)
def test_majority_vote(monkeypatch, labels, expected):
    n = len(labels)
    monkeypatch.setattr(service, "split_segments", lambda signal: [np.zeros(10)] * n)
    monkeypatch.setattr(service, "segment_features", lambda segment: np.zeros(30))
    result = service.predict_signal(np.zeros(SAMPLE_RATE), FakeSession(labels, [0.2] * n))
    assert result["label"] == expected
    assert result["probability_abnormal"] == pytest.approx(0.2)
