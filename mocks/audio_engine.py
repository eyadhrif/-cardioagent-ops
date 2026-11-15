"""MOCK of Service 3 (audio-engine). Returns a FAKE prediction so the gateway flow can be tested.

The real service would resample to 2000 Hz, extract MFCC and run the ONNX model.
"""

from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="Mock audio-engine")


@app.get("/health")
def health():
    return {"status": "ok", "service": "audio-engine (mock)"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    # Fake but deterministic: the same file always gives the same "probability".
    probability = round((len(content) % 100) / 100, 2)
    return {
        "label": "abnormal" if probability >= 0.5 else "normal",
        "probability_abnormal": probability,
        "model_name": "mock-model (not a real prediction)",
    }
