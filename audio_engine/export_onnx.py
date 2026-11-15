"""Export the selected SVM to ONNX (audio_engine/cardio_model.onnx), as the brief asks.

train_svm.py compares the kernels and selects rbf (best mean recall). Here we train that pipeline on ALL
the segments with probability=True (so the service can return a probability), convert it to ONNX, check
that ONNX Runtime gives the same answers as scikit-learn, and log the file to MLflow.

Run:  uv run python -m audio_engine.export_onnx
"""

# onnxruntime FIRST: on Windows, importing it after pandas/mlflow crashes (two different C++ runtime DLLs)
import onnxruntime as ort  # isort: skip

import argparse
from pathlib import Path

import mlflow
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from audio_engine.features import build_dataset, feature_names
from audio_engine.train_svm import KERNELS, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RANDOM_STATE, make_pipeline

BEST_KERNEL = "rbf"  # chosen by train_svm.py: best mean recall on the abnormal class (see README)
MODEL_PATH = Path(__file__).parent / "cardio_model.onnx"
INPUT_NAME = "features"  # name of the ONNX input: a float32 matrix (n_segments, 30)


def train_final_model(X, y, kernel: str = BEST_KERNEL) -> Pipeline:
    """Scaler + SVM trained on all the data. probability=True adds Platt scaling (a calibrated probability)."""
    params = {**KERNELS[kernel], "probability": True, "random_state": RANDOM_STATE}
    return make_pipeline(params).fit(X, y)


def to_onnx_bytes(model: Pipeline, n_features: int) -> bytes:
    """Convert the whole pipeline (scaler included) to ONNX.

    Outputs: "label" (0/1 per segment) and "probabilities" (n_segments, 2). zipmap=False keeps a plain matrix.
    """
    onnx_model = convert_sklearn(
        model,
        initial_types=[(INPUT_NAME, FloatTensorType([None, n_features]))],
        options={id(model.named_steps["svm"]): {"zipmap": False}},
    )
    return onnx_model.SerializeToString()


def compare_with_sklearn(onnx_bytes: bytes, model: Pipeline, X) -> dict:
    """ONNX works in float32, scikit-learn in float64: answers must still be (almost) identical."""
    session = ort.InferenceSession(onnx_bytes)
    labels, probabilities = session.run(None, {INPUT_NAME: np.asarray(X, dtype=np.float32)})
    return {
        "label_agreement": float(np.mean(labels == model.predict(X))),
        "max_probability_diff": float(np.max(np.abs(probabilities - model.predict_proba(X)))),
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Export the selected SVM to ONNX")
    parser.add_argument("--rebuild", action="store_true", help="extract the features again (ignore the cache)")
    args = parser.parse_args(argv)

    df = build_dataset(rebuild=args.rebuild)
    X, y = df[feature_names()], df["label"]
    print(f"Training the final {BEST_KERNEL} SVM on {len(df)} segments (probability=True)...")
    model = train_final_model(X, y)

    onnx_bytes = to_onnx_bytes(model, X.shape[1])
    MODEL_PATH.write_bytes(onnx_bytes)
    check = compare_with_sklearn(onnx_bytes, model, X)
    print(f"Saved {MODEL_PATH} ({len(onnx_bytes) / 1024:.0f} KB), check vs scikit-learn: {check}")

    # Traceability: which data and settings produced this exact file
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"export-onnx-{BEST_KERNEL}"):
        mlflow.log_params({**KERNELS[BEST_KERNEL], "probability": True, "n_segments": len(df),
                           "n_files": df["file"].nunique(), "n_features": X.shape[1]})
        mlflow.log_metrics({**check, "n_support_vectors": int(model.named_steps["svm"].n_support_.sum())})
        mlflow.log_artifact(str(MODEL_PATH))
    return check


if __name__ == "__main__":
    main()
