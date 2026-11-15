"""Tests of TP3 (audio_engine): features, metrics, grouped folds, and training with MLflow mocked.

Only tiny synthetic .wav files in tmp_path are used: no real dataset, no MLflow server.
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from audio_engine import features, train_svm
from audio_engine.features import SAMPLE_RATE, SEGMENT_SECONDS

SEG_LEN = int(SEGMENT_SECONDS * SAMPLE_RATE)  # 10000 samples


def write_wav(path, seconds: float, sample_rate: int = 4000, freq: float = 50.0):
    """Write a sine wave (like a low heart sound) + a little noise as a 16-bit wav."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    rng = np.random.default_rng(int(freq))
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(len(t))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate, subtype="PCM_16")
    return path


# ---------- features.py ----------

def test_load_audio_resamples_to_2000_hz_and_normalises(tmp_path):
    path = write_wav(tmp_path / "a.wav", seconds=3, sample_rate=4000)
    signal = features.load_audio(path)
    assert len(signal) == 3 * SAMPLE_RATE           # 12000 samples at 4000 Hz -> 6000 at 2000 Hz
    assert np.max(np.abs(signal)) == pytest.approx(1.0)


def test_load_audio_silent_file_does_not_crash(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(4000), 2000, subtype="PCM_16")
    signal = features.load_audio(path)
    assert not np.isnan(signal).any()


@pytest.mark.parametrize(
    "seconds, expected_segments",
    [
        (0.76, 1),   # shorter than 5 s -> one padded segment
        (5.0, 1),    # exactly 5 s
        (6.0, 1),    # 5 s + 1 s left -> the 1 s piece is dropped
        (7.5, 2),    # 5 s + 2.5 s left -> the 2.5 s piece is padded
        (15.0, 3),
    ],
)
def test_split_segments_count_and_length(seconds, expected_segments):
    signal = np.ones(int(seconds * SAMPLE_RATE), dtype=np.float32)
    segments = features.split_segments(signal)
    assert len(segments) == expected_segments
    assert all(len(s) == SEG_LEN for s in segments)


def test_segment_features_shape_and_no_nan():
    segment = np.sin(np.linspace(0, 200, SEG_LEN)).astype(np.float32)
    vector = features.segment_features(segment)
    assert vector.shape == (30,)
    assert len(features.feature_names()) == 30
    assert not np.isnan(vector).any()


def make_dataset_folder(root):
    """6 normal + 6 abnormal small recordings of different lengths."""
    for i in range(6):
        write_wav(root / "normal" / f"n{i}.wav", seconds=4 + 2 * i, freq=40 + i)
        write_wav(root / "abnormal" / f"a{i}.wav", seconds=4 + 2 * i, freq=200 + 10 * i)


def test_build_dataset_and_cache(tmp_path):
    data_dir, cache = tmp_path / "train", tmp_path / "cache" / "features.csv"
    make_dataset_folder(data_dir)

    df = features.build_dataset(data_dir, cache)
    assert cache.exists()
    assert list(df.columns[:3]) == ["file", "segment", "label"]
    assert df.shape[1] == 3 + 30
    assert set(df["label"]) == {0, 1}
    assert df["file"].nunique() == 12
    assert not df[features.feature_names()].isna().any().any()

    # Second call reads the cache (same content), --rebuild extracts again
    assert features.build_dataset(data_dir, cache).shape == df.shape
    assert features.build_dataset(data_dir, cache, rebuild=True).shape == df.shape


# ---------- train_svm.py ----------

def test_compute_metrics_course_example():
    # Example of the course slide: TP=2, FP=2, FN=1, TN=2
    y_true = [0, 0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 1, 1, 0]
    m = train_svm.compute_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 7)
    assert m["specificity"] == pytest.approx(2 / 4)
    assert m["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert math.isnan(m["roc_auc"])  # no scores given


def test_compute_metrics_roc_auc_and_zero_division():
    m = train_svm.compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], scores=[-2, -1, 1, 2])
    assert m["roc_auc"] == 1.0 and m["recall"] == 1.0
    m = train_svm.compute_metrics([0, 0], [0, 0])  # no positive at all: no division by zero
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_folds_never_share_a_file():
    # 30 files, each cut into 1 to 4 segments
    rng = np.random.default_rng(0)
    files, labels = [], []
    for i in range(30):
        n_segments = rng.integers(1, 5)
        files += [f"file{i}"] * n_segments
        labels += [int(i % 3 == 0)] * n_segments
    groups, y = pd.Series(files), pd.Series(labels)

    folds = train_svm.make_folds(y, groups)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[test_idx])
        assert y.iloc[test_idx].nunique() == 2  # stratified: both classes in each test fold


def test_figures_are_created():
    folds = [{"accuracy": 0.8, "recall": 0.9}, {"accuracy": 0.7, "recall": float("nan")}]
    assert len(train_svm.metrics_figure(folds, "t").axes) == 1
    curves = [([0, 0, 1, 1], [-1.0, 0.2, 0.5, 2.0]), ([0, 1, 0, 1], [-2.0, 1.0, 0.1, 0.3])]
    assert len(train_svm.roc_figure(curves, "t").axes[0].lines) == 3  # 2 folds + chance line
    summary = {f"{m}_{s}": 0.5 for m in ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
               for s in ["mean", "std"]}
    fig = train_svm.comparison_figure({"linear": ("id1", summary), "rbf": ("id2", summary)})
    assert len(fig.axes[0].patches) == 12  # 6 metrics x 2 kernels
    train_svm.plt.close("all")


def test_pipeline_scaler_is_inside():
    pipe = train_svm.make_pipeline({"kernel": "rbf", "C": 1.0, "gamma": "scale"})
    assert [name for name, _ in pipe.steps] == ["scaler", "svm"]
    assert pipe.named_steps["svm"].class_weight == "balanced"


def fake_features(n_files: int = 24) -> pd.DataFrame:
    """A small feature table where abnormal files have shifted features (easy to learn)."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n_files):
        label = int(i % 3 == 0)
        for seg in range(2):
            values = rng.normal(loc=2.0 * label, size=30)
            rows.append({"file": f"f{i}.wav", "segment": seg, "label": label,
                         **dict(zip(features.feature_names(), values))})
    return pd.DataFrame(rows)


def test_main_trains_both_kernels_with_mlflow_mocked(monkeypatch):
    fake_mlflow = MagicMock()
    monkeypatch.setattr(train_svm, "mlflow", fake_mlflow)
    monkeypatch.setattr(train_svm, "build_dataset", lambda rebuild=False: fake_features())

    best = train_svm.main([])

    assert best in train_svm.KERNELS
    # 2 parent runs + 2 x 3 nested child runs
    assert fake_mlflow.start_run.call_count == 2 + 2 * 3
    nested = [c for c in fake_mlflow.start_run.call_args_list if c.kwargs.get("nested")]
    assert len(nested) == 6
    # 6 folds x (confusion matrix + ROC) + 2 kernels x (metrics bars + ROC curves)
    assert fake_mlflow.log_figure.call_count == 6 * 2 + 2 * 2
    figure_names = {c.args[1] for c in fake_mlflow.log_figure.call_args_list}
    assert figure_names == {"confusion_matrix.png", "roc_curve.png", "metrics_per_fold.png", "roc_curves.png"}
    # linear vs rbf comparison chart added to both parent runs
    assert fake_mlflow.MlflowClient.return_value.log_figure.call_count == 2
    assert fake_mlflow.sklearn.log_model.call_count == 2        # one final model per kernel
    fake_mlflow.set_experiment.assert_called_once_with(train_svm.MLFLOW_EXPERIMENT)
    model_uri, name = fake_mlflow.register_model.call_args.args
    assert name == "cardio-svm" and model_uri.endswith("/model")
