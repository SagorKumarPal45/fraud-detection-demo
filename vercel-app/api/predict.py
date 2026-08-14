from http.server import BaseHTTPRequestHandler
import json
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "model-lite"

REQUIRED = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount",
]

MODELS = {
    "logistic_regression": "logistic_regression.joblib",
    "decision_tree": "decision_tree.joblib",
    "mlp": "mlp.joblib",
    "svc": "svc.joblib",
}

_cache = {}


def get_artifact(name):
    if name not in _cache:
        _cache[name] = joblib.load(MODEL_DIR / name)
    return _cache[name]


def preprocess(df):
    scaler = get_artifact("robust_scaler.joblib")
    pca = get_artifact("pca_transformer.joblib")
    selected = get_artifact("selected_features.joblib")

    work = df[REQUIRED].copy()
    work["scaled_amount"] = scaler.transform(work["Amount"].values.reshape(-1, 1))
    work["scaled_time"] = scaler.transform(work["Time"].values.reshape(-1, 1))
    work = work.drop(["Time", "Amount"], axis=1)
    sa, st = work["scaled_amount"], work["scaled_time"]
    work = work.drop(["scaled_amount", "scaled_time"], axis=1)
    work.insert(0, "scaled_amount", sa)
    work.insert(1, "scaled_time", st)
    return pca.transform(work[selected])


def fraud_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].tolist()
    scores = model.decision_function(x)
    return (1 / (1 + np.exp(-scores))).tolist()


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            csv_text = body.get("csv", "")
            model_key = body.get("model", "logistic_regression")
            if model_key not in MODELS:
                raise ValueError(f"Unknown model: {model_key}")

            df = pd.read_csv(io.StringIO(csv_text))
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {', '.join(missing)}")

            model = get_artifact(MODELS[model_key])
            x = preprocess(df)
            preds = model.predict(x).astype(int).tolist()
            probas = fraud_proba(model, x)

            rows = []
            for i, (p, pr) in enumerate(zip(preds, probas), start=1):
                row = {
                    "row": i,
                    "prediction": p,
                    "fraud_probability": round(pr, 4),
                    "result_label": "FRAUD" if p == 1 else "NORMAL",
                }
                if "Class" in df.columns:
                    actual = int(df.iloc[i - 1]["Class"])
                    row["actual_class"] = actual
                    row["correct"] = p == actual
                rows.append(row)

            flagged = sum(1 for r in rows if r["prediction"] == 1)
            payload = {
                "ok": True,
                "summary": {
                    "total": len(rows),
                    "flagged_fraud": flagged,
                    "flagged_normal": len(rows) - flagged,
                },
                "results": rows,
            }
            self._json(200, payload)
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})

    def _json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
