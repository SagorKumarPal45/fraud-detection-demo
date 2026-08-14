"use client";

import { useState } from "react";

type ResultRow = {
  row: number;
  prediction: number;
  fraud_probability: number;
  result_label: string;
  actual_class?: number;
  correct?: boolean;
};

type ApiResponse = {
  ok: boolean;
  error?: string;
  summary?: { total: number; flagged_fraud: number; flagged_normal: number };
  results?: ResultRow[];
};

const MODELS = [
  { id: "logistic_regression", label: "Logistic Regression" },
  { id: "decision_tree", label: "Decision Tree" },
  { id: "mlp", label: "Neural Network (MLP)" },
  { id: "svc", label: "Support Vector Machine" },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState("logistic_regression");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<ApiResponse | null>(null);

  async function runDetection() {
    if (!file) {
      setError("Please upload a CSV file first.");
      return;
    }
    setLoading(true);
    setError("");
    setData(null);
    try {
      const csv = await file.text();
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ csv, model }),
      });
      const json: ApiResponse = await res.json();
      if (!json.ok) throw new Error(json.error || "Prediction failed");
      setData(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Credit Card Fraud Detection</h1>
        <p>Group 4 – Threat Detection Project · Upload CSV to test our ML models</p>
      </header>

      <div className="card">
        <h2>1. Download sample data</h2>
        <div className="row">
          <a className="btn-secondary" href="/sample_upload_data.csv" download>
            Sample CSV (no labels)
          </a>
          <a className="btn-secondary" href="/sample_test_data.csv" download>
            Sample CSV (with Class)
          </a>
        </div>
        <p className="info" style={{ marginTop: "1rem" }}>
          Use the same format as creditcard.csv: Time, V1–V28, Amount
        </p>
      </div>

      <div className="card">
        <h2>2. Upload &amp; test</h2>
        <div className="row">
          <label className="file-label">
            Choose CSV
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          {file && <span className="file-name">{file.name}</span>}
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          <button className="btn-primary" onClick={runDetection} disabled={loading}>
            {loading ? "Running…" : "Run detection"}
          </button>
        </div>
        <p className="info" style={{ marginTop: "1rem" }}>
          Vercel deploy uses lightweight models only.
          <span className="badge">RF / KNN / Stacking need Streamlit</span>
        </p>
        {error && <div className="error">{error}</div>}
      </div>

      {data?.summary && data.results && (
        <div className="card">
          <h2>3. Results</h2>
          <div className="metrics">
            <div className="metric">
              <div className="value">{data.summary.total}</div>
              <div className="label">Transactions</div>
            </div>
            <div className="metric">
              <div className="value">{data.summary.flagged_fraud}</div>
              <div className="label">Flagged fraud</div>
            </div>
            <div className="metric">
              <div className="value">{data.summary.flagged_normal}</div>
              <div className="label">Flagged normal</div>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Row</th>
                  <th>Prediction</th>
                  <th>Fraud prob.</th>
                  {"actual_class" in data.results[0] && <th>Actual</th>}
                  {"correct" in data.results[0] && <th>Match</th>}
                </tr>
              </thead>
              <tbody>
                {data.results.map((r) => (
                  <tr key={r.row}>
                    <td>{r.row}</td>
                    <td className={r.prediction === 1 ? "fraud" : "normal"}>
                      {r.result_label}
                    </td>
                    <td>{(r.fraud_probability * 100).toFixed(1)}%</td>
                    {"actual_class" in r && (
                      <td className={r.actual_class === 1 ? "fraud" : "normal"}>
                        {r.actual_class === 1 ? "FRAUD" : "NORMAL"}
                      </td>
                    )}
                    {"correct" in r && (
                      <td>{r.correct ? "✓" : "✗"}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
