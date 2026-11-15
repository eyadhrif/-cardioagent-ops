"""TP3 step 2: SVM (linear vs rbf kernel) with 3 grouped folds, tracked in MLflow.

MLflow layout (experiment "TP3-SVM"):
    svm-linear  (parent run: mean/std of the metrics + final model)
        svm-linear-fold1, fold2, fold3   (child runs: metrics + confusion matrix)
    svm-rbf     (same)
The kernel with the best mean recall is registered as the model "cardio-svm".

Run:  uv run python -m audio_engine.train_svm [--rebuild]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # no screen in Docker: draw figures in memory only
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audio_engine.features import build_dataset, feature_names

# Settings from environment variables (docker-compose.yml sets them).
# Default: a local SQLite file, so the script also works without the mlflow-server.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "TP3-SVM")
REGISTERED_MODEL_NAME = "cardio-svm"

N_SPLITS = 3
RANDOM_STATE = 42

# The two kernels of TP3 (kernel trick: rbf maps the data to a space where it can be separated linearly)
KERNELS = {
    "linear": {"kernel": "linear", "C": 1.0},
    "rbf": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
}


def make_pipeline(kernel_params: dict) -> Pipeline:
    """Scaler + SVM in ONE pipeline: the scaler is fitted on the training fold only (no leakage)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        # class_weight="balanced": errors on the rare class (abnormal, 23%) cost more
        ("svm", SVC(class_weight="balanced", cache_size=500, **kernel_params)),
    ])


def compute_metrics(y_true, y_pred, scores=None) -> dict:
    """The 6 metrics of the course. The positive class is 1 = abnormal."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0     # among predicted abnormal, how many are abnormal
    recall = tp / (tp + fn) if tp + fn else 0.0        # among real abnormal, how many we found (sensitivity)
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,  # among real normal, how many we called normal
    }
    # ROC-AUC needs a continuous score (distance to the SVM hyperplane) and both classes present
    if scores is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
    else:
        metrics["roc_auc"] = float("nan")
    return {name: float(value) for name, value in metrics.items()}


def make_folds(y, groups) -> list:
    """3 stratified folds grouped by file: all segments of one recording stay on the same side."""
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def confusion_figure(y_true, y_pred, title: str):
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=["normal", "abnormal"], cmap="Blues"
    )
    display.ax_.set_title(title)
    return display.figure_


# ---------- Figures logged in MLflow (Artifacts tab), so the metrics are also seen as graphics ----------

def metrics_figure(fold_metrics: list, title: str):
    """Bars = mean of each metric over the folds, black line = std, dots = the value of each fold."""
    names = list(fold_metrics[0])
    values = np.array([[m[name] for name in names] for m in fold_metrics])  # shape (n_folds, n_metrics)
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, np.nanmean(values, axis=0), yerr=np.nanstd(values, axis=0), capsize=4, alpha=0.6, label="mean ± std")
    for k, fold_values in enumerate(values, start=1):
        ax.scatter(x, fold_values, s=20, zorder=3, label=f"fold {k}")
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))  # outside the plot, on the right
    ax.set_title(title)
    fig.tight_layout()
    return fig


def roc_figure(curves: list, title: str):
    """ROC curve (true positive rate vs false positive rate) of each fold. curves = [(y_true, scores), ...]"""
    fig, ax = plt.subplots(figsize=(5, 5))
    for k, (y_true, scores) in enumerate(curves, start=1):
        RocCurveDisplay.from_predictions(y_true, scores, name=f"fold {k}", ax=ax)
    ax.plot([0, 1], [0, 1], "k:", label="chance (AUC = 0.5)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    return fig


def comparison_figure(results: dict):
    """One group of bars per metric, one bar per kernel (mean ± std over the folds)."""
    metrics = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, (_, summary)) in enumerate(results.items()):
        means = [summary[f"{m}_mean"] for m in metrics]
        stds = [summary[f"{m}_std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=f"{name} kernel")
    ax.set_xticks(x + width * (len(results) - 1) / 2, metrics)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))  # outside the plot, on the right
    ax.set_title("SVM kernels compared (mean ± std over 3 folds)")
    fig.tight_layout()
    return fig


def log_and_close(fig, path: str) -> None:
    mlflow.log_figure(fig, path)  # saved as a .png in the Artifacts tab of the current run
    plt.close(fig)


def run_kernel(name: str, params: dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: list):
    """One parent run (summary + final model) with one nested child run per fold."""
    with mlflow.start_run(run_name=f"svm-{name}") as parent:
        mlflow.log_params({
            **params,
            "class_weight": "balanced",
            "cv": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
            "n_segments": len(X),
            "n_files": groups.nunique(),
            "n_features": X.shape[1],
        })

        fold_metrics = []
        roc_curves = []  # (y_true, scores) of each fold, for the ROC figure of the parent run
        for k, (train_idx, test_idx) in enumerate(folds, start=1):
            with mlflow.start_run(run_name=f"svm-{name}-fold{k}", nested=True):
                model = make_pipeline(params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])

                y_test = y.iloc[test_idx]
                y_pred = model.predict(X.iloc[test_idx])
                scores = model.decision_function(X.iloc[test_idx])  # signed distance to the hyperplane
                metrics = compute_metrics(y_test, y_pred, scores)

                mlflow.log_params({
                    **params,
                    "fold": k,
                    "n_train_segments": len(train_idx),
                    "n_test_segments": len(test_idx),
                    "n_train_files": groups.iloc[train_idx].nunique(),
                    "n_test_files": groups.iloc[test_idx].nunique(),
                })
                mlflow.log_metrics(metrics)
                log_and_close(confusion_figure(y_test, y_pred, f"{name} kernel - fold {k}"), "confusion_matrix.png")
                log_and_close(roc_figure([(y_test, scores)], f"ROC - {name} kernel - fold {k}"), "roc_curve.png")

            fold_metrics.append(metrics)
            roc_curves.append((y_test, scores))
            print(f"  {name} fold {k}: " + ", ".join(f"{m}={v:.3f}" for m, v in metrics.items()))

        # Mean and std over the 3 folds (std = how stable the model is from one split to another)
        summary = {}
        for metric in fold_metrics[0]:
            values = [m[metric] for m in fold_metrics]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        mlflow.log_metrics(summary)
        # Same names as the child runs (value = mean), so every row of the MLflow table fills the same columns
        mlflow.log_metrics({metric: summary[f"{metric}_mean"] for metric in fold_metrics[0]})
        # One point per fold (step = fold number): the "Chart" tab of the parent run shows fold 1 -> 2 -> 3
        for k, metrics in enumerate(fold_metrics, start=1):
            mlflow.log_metrics({f"fold_{m}": v for m, v in metrics.items()}, step=k)

        # The same numbers as graphics
        log_and_close(metrics_figure(fold_metrics, f"{name} kernel - metrics over {N_SPLITS} folds"), "metrics_per_fold.png")
        log_and_close(roc_figure(roc_curves, f"ROC curves - {name} kernel"), "roc_curves.png")

        # Final model: trained on ALL the data, this is the one we would deploy
        final_model = make_pipeline(params).fit(X, y)
        signature = infer_signature(X, final_model.predict(X))  # input = 30 named float columns, output = 0/1
        mlflow.sklearn.log_model(
            final_model, artifact_path="model", signature=signature, input_example=X.head(5)
        )

    return parent.info.run_id, summary


def print_summary(results: dict) -> None:
    print("\n=== Summary (mean ± std over 3 folds) ===")
    metrics = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
    print(f"{'kernel':<8}" + "".join(f"{m:>18}" for m in metrics))
    for name, (_, summary) in results.items():
        cells = "".join(f"{summary[m + '_mean']:>11.3f} ±{summary[m + '_std']:.3f}" for m in metrics)
        print(f"{name:<8}{cells}")


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description="TP3: SVM linear vs rbf, tracked in MLflow")
    parser.add_argument("--rebuild", action="store_true", help="extract the features again (ignore the cache)")
    args = parser.parse_args(argv)

    df = build_dataset(rebuild=args.rebuild)
    X = df[feature_names()]
    y = df["label"]
    groups = df["file"]
    print(f"{len(df)} segments, {groups.nunique()} files, abnormal segments = {y.mean():.1%}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    folds = make_folds(y, groups)  # the SAME folds for both kernels: a fair comparison
    results = {name: run_kernel(name, params, X, y, groups, folds) for name, params in KERNELS.items()}
    print_summary(results)

    # Comparison chart linear vs rbf, added to both parent runs (the runs are closed, so we use the client)
    fig = comparison_figure(results)
    client = mlflow.MlflowClient()
    for run_id, _ in results.values():
        client.log_figure(run_id, fig, "kernel_comparison.png")
    plt.close(fig)

    # Selection rule of the brief: best recall on the pathological class
    # (a missed abnormal heart = false negative = the worst error in screening)
    best = max(results, key=lambda name: results[name][1]["recall_mean"])
    run_id = results[best][0]
    mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    print(f"\nBest kernel by mean recall: {best} -> registered as '{REGISTERED_MODEL_NAME}'")
    return best


if __name__ == "__main__":
    main()
