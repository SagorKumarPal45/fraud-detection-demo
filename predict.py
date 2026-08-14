"""Credit card fraud detection inference pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

REQUIRED_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount",
]

AVAILABLE_MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Support Vector Machine": "svc.joblib",
    "Neural Network (MLP)": "mlp.joblib",
}

# XGBoost and Stacking Classifier files are corrupted in the saved bundle.
UNAVAILABLE_MODELS = ["XGBoost", "Stacking Classifier"]

_artifacts: dict | None = None
_model_cache: dict = {}


def _load_artifacts() -> dict:
    global _artifacts
    if _artifacts is None:
        _artifacts = {
            "scaler": joblib.load(MODEL_DIR / "robust_scaler.joblib"),
            "pca": joblib.load(MODEL_DIR / "pca_transformer.joblib"),
            "selected_features": joblib.load(MODEL_DIR / "selected_features.joblib"),
        }
    return _artifacts


def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if model_name not in _model_cache:
        path = MODEL_DIR / AVAILABLE_MODELS[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        _model_cache[model_name] = joblib.load(path)
    return _model_cache[model_name]


def validate_input(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + ". Expected the same format as creditcard.csv (Time, V1–V28, Amount)."
        )
    if len(df) == 0:
        raise ValueError("Uploaded CSV has no data rows.")


def preprocess(df: pd.DataFrame) -> np.ndarray:
    validate_input(df)
    artifacts = _load_artifacts()
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]
    selected_features = artifacts["selected_features"]

    work = df[REQUIRED_COLUMNS].copy()
    work["scaled_amount"] = scaler.transform(work["Amount"].values.reshape(-1, 1))
    work["scaled_time"] = scaler.transform(work["Time"].values.reshape(-1, 1))
    work = work.drop(["Time", "Amount"], axis=1)

    scaled_amount = work["scaled_amount"]
    scaled_time = work["scaled_time"]
    work = work.drop(["scaled_amount", "scaled_time"], axis=1)
    work.insert(0, "scaled_amount", scaled_amount)
    work.insert(1, "scaled_time", scaled_time)

    x_selected = work[selected_features]
    return pca.transform(x_selected)


def get_fraud_probability(model, x_pca: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_pca)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_pca)
        return 1 / (1 + np.exp(-scores))
    preds = model.predict(x_pca)
    return preds.astype(float)


def predict_fraud(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model = load_model(model_name)
    x_pca = preprocess(df)
    predictions = model.predict(x_pca)
    probabilities = get_fraud_probability(model, x_pca)

    results = pd.DataFrame(
        {
            "row": range(1, len(df) + 1),
            "prediction": predictions.astype(int),
            "fraud_probability": np.round(probabilities, 4),
            "result_label": ["FRAUD" if p == 1 else "NORMAL" for p in predictions],
        }
    )

    if "Class" in df.columns:
        actual = df["Class"].astype(int).values
        results["actual_class"] = actual
        results["actual_label"] = ["FRAUD" if a == 1 else "NORMAL" for a in actual]
        results["correct"] = results["prediction"] == results["actual_class"]

    return results


def summarize_results(results: pd.DataFrame) -> dict:
    summary = {
        "total_transactions": len(results),
        "flagged_fraud": int((results["prediction"] == 1).sum()),
        "flagged_normal": int((results["prediction"] == 0).sum()),
        "avg_fraud_probability": float(results["fraud_probability"].mean()),
    }
    if "actual_class" in results.columns:
        fraud_mask = results["actual_class"] == 1
        normal_mask = results["actual_class"] == 0
        summary["actual_fraud"] = int(fraud_mask.sum())
        summary["actual_normal"] = int(normal_mask.sum())
        summary["accuracy"] = float(results["correct"].mean())
        if fraud_mask.any():
            summary["fraud_recall"] = float(
                (results.loc[fraud_mask, "prediction"] == 1).mean()
            )
    return summary
