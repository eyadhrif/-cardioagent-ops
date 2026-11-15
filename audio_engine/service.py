"""Service 3: audio-engine HTTP API (replaces mocks/audio_engine.py, same contract).

POST /predict (multipart field "file" = .wav) -> {"label", "probability_abnormal", "model_name"}
Same preprocessing as training: 2000 Hz, peak normalisation, 5 s segments, 30 features per segment.
Inference with ONNX Runtime on audio_engine/cardio_model.onnx (made by audio_engine/export_onnx.py).

Run:  uv run uvicorn audio_engine.service:app --port 8001
"""

# onnxruntime FIRST: on Windows, importing it after pandas crashes (two different C++ runtime DLLs)
import onnxruntime as ort  # isort: skip

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from audio_engine.features import load_audio, segment_features, split_segments

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "cardio_model.onnx"))
MODEL_NAME = "cardio-svm-rbf (ONNX)"

app = FastAPI(title="CardioAgent-Ops audio-engine", version="1.0.0")


@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    """Load the ONNX model once, at the first request, then keep it in memory."""
    return ort.InferenceSession(str(MODEL_PATH))


def predict_signal(signal: np.ndarray, session: ort.InferenceSession) -> dict:
    """One prediction per 5 s segment, then one answer for the whole recording."""
    features = np.stack([segment_features(s) for s in split_segments(signal)]).astype(np.float32)
    input_name = session.get_inputs()[0].name
    labels, probabilities = session.run(None, {input_name: features})

    # Majority vote of the SVM labels, a tie counts as abnormal: the SVM was trained with
    # class_weight="balanced" to favour recall, and a missed sick heart is the worst error.
    abnormal_votes = float(np.mean(labels))
    return {
        "label": "abnormal" if abnormal_votes >= 0.5 else "normal",
        "probability_abnormal": round(float(np.mean(probabilities[:, 1])), 4),
        "model_name": MODEL_NAME,
        "n_segments": len(features),
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "audio-engine", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    try:
        signal = load_audio(io.BytesIO(content))  # librosa reads from memory: no temporary file
    except Exception:
        raise HTTPException(status_code=400, detail="The file is not a readable audio file")
    if signal.size == 0:
        raise HTTPException(status_code=400, detail="The audio file contains no samples")
    return predict_signal(signal, get_session())
