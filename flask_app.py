"""Flask web app for credit card fraud detection (works reliably on Windows)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request, send_file

from person_data import get_labeled_dataset, identify_person, load_persons
from predict import AVAILABLE_MODELS, UNAVAILABLE_MODELS, predict_fraud, summarize_results

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "person_data" / "datasets"
app = Flask(__name__)

STYLES = """
    :root {
      --bg: #070b14; --surface: #111827; --surface2: #1a2332; --border: #243044;
      --text: #f1f5f9; --muted: #94a3b8; --accent: #3b82f6; --accent2: #6366f1;
      --green: #10b981; --red: #ef4444; --nav-h: 68px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body { font-family: 'Inter', Segoe UI, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; line-height: 1.6; }
    .navbar {
      position: fixed; top: 0; left: 0; right: 0; z-index: 1000; height: var(--nav-h);
      background: rgba(7,11,20,0.85); backdrop-filter: blur(16px); border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; padding: 0 2rem;
    }
    .nav-brand { display: flex; align-items: center; gap: 0.6rem; text-decoration: none; color: var(--text); font-weight: 800; font-size: 1.15rem; }
    .nav-brand .logo { width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, var(--accent), var(--accent2)); display: flex; align-items: center; justify-content: center; }
    .nav-links { display: flex; gap: 0.25rem; list-style: none; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; font-weight: 500; padding: 0.5rem 1rem; border-radius: 8px; transition: all 0.2s; }
    .nav-links a:hover, .nav-links a.active { color: var(--text); background: rgba(59,130,246,0.12); }
    .nav-cta { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: white !important; padding: 0.5rem 1.1rem !important; border-radius: 8px; }
    .page { padding-top: var(--nav-h); }
    .container { max-width: 1140px; margin: 0 auto; padding: 0 1.5rem; }
    section { padding: 4rem 0; }
    .section-label { display: inline-block; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); background: rgba(59,130,246,0.1); padding: 0.3rem 0.75rem; border-radius: 999px; margin-bottom: 0.75rem; }
    .section-title { font-size: 1.75rem; font-weight: 800; margin-bottom: 0.5rem; }
    .section-desc { color: var(--muted); max-width: 560px; margin-bottom: 2rem; }
    .hero { padding: 5rem 0 3rem; background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(59,130,246,0.18), transparent); text-align: center; }
    .hero h1 { font-size: clamp(2rem, 5vw, 3rem); font-weight: 800; background: linear-gradient(135deg, #fff 30%, #93c5fd); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 540px; margin: 0 auto 2rem; }
    .hero-btns { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
    .btn { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.75rem 1.5rem; border-radius: 10px; font-weight: 600; font-size: 0.95rem; text-decoration: none; border: none; cursor: pointer; transition: all 0.2s; }
    .btn-primary { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: white; }
    .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(59,130,246,0.35); }
    .btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .btn-outline:hover { border-color: var(--accent); color: var(--accent); }
    .btn-green { background: linear-gradient(135deg, #059669, var(--green)); color: white; width: 100%; justify-content: center; }
    .btn-green:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(16,185,129,0.3); }
    .hero-stats { display: flex; gap: 2.5rem; justify-content: center; flex-wrap: wrap; margin-top: 2.5rem; padding-top: 2rem; border-top: 1px solid var(--border); }
    .hero-stat b { display: block; font-size: 1.6rem; font-weight: 800; color: var(--accent); }
    .hero-stat span { font-size: 0.8rem; color: var(--muted); }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; }
    .test-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: start; }
    @media (max-width: 768px) { .test-grid { grid-template-columns: 1fr; } .nav-links a:not(.nav-cta) { display: none; } }
    .form-group { margin-bottom: 1.25rem; }
    .form-group label { display: block; font-size: 0.85rem; font-weight: 600; color: var(--muted); margin-bottom: 0.5rem; }
    select, .file-drop { width: 100%; padding: 0.75rem 1rem; border-radius: 10px; background: var(--surface2); border: 1px solid var(--border); color: var(--text); font-size: 0.9rem; font-family: inherit; }
    select:focus, .file-drop:focus-within { outline: none; border-color: var(--accent); }
    .file-drop { border-style: dashed; text-align: center; cursor: pointer; transition: border-color 0.2s; }
    .file-drop:hover { border-color: var(--accent); background: rgba(59,130,246,0.05); }
    .file-drop input { display: none; }
    .file-drop .icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .file-drop p { color: var(--muted); font-size: 0.85rem; }
    .file-name-display { color: var(--green); font-size: 0.85rem; margin-top: 0.5rem; font-weight: 600; }
    .steps-list { list-style: none; }
    .steps-list li { display: flex; gap: 1rem; padding: 1rem 0; border-bottom: 1px solid var(--border); }
    .steps-list li:last-child { border-bottom: none; }
    .step-num { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; background: rgba(59,130,246,0.15); color: var(--accent); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; }
    .step-text b { display: block; font-size: 0.9rem; margin-bottom: 0.15rem; }
    .step-text span { color: var(--muted); font-size: 0.82rem; }
    .result-hero { background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(99,102,241,0.15)); border: 1px solid rgba(59,130,246,0.3); border-radius: 16px; padding: 1.75rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap; }
    .avatar { width: 48px; height: 48px; border-radius: 50%; flex-shrink: 0; background: linear-gradient(135deg, var(--accent), var(--accent2)); display: flex; align-items: center; justify-content: center; font-weight: 700; color: white; }
    .result-hero .avatar { width: 56px; height: 56px; font-size: 1.1rem; }
    .result-hero h2 { font-size: 1.4rem; font-weight: 800; }
    .result-hero p { color: var(--muted); font-size: 0.9rem; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; }
    .metric { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; text-align: center; }
    .metric b { display: block; font-size: 2rem; font-weight: 800; color: var(--accent); }
    .metric.danger b { color: var(--red); }
    .metric.success b { color: var(--green); }
    .metric span { font-size: 0.78rem; color: var(--muted); margin-top: 0.25rem; display: block; }
    .insight { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); border-radius: 10px; padding: 1rem 1.25rem; margin-top: 1.25rem; font-size: 0.9rem; color: #a7f3d0; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th { color: var(--muted); font-weight: 600; padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
    td { padding: 0.55rem 0.75rem; border-bottom: 1px solid rgba(36,48,68,0.6); }
    .fraud { color: #f87171; font-weight: 700; }
    .normal { color: #4ade80; font-weight: 700; }
    .table-wrap { max-height: 400px; overflow: auto; border-radius: 10px; border: 1px solid var(--border); margin-top: 1rem; }
    .pipeline { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
    .pipe-step { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; text-align: center; }
    .pipe-step .icon { font-size: 1.75rem; margin-bottom: 0.5rem; }
    .pipe-step b { display: block; font-size: 0.9rem; margin-bottom: 0.25rem; }
    .pipe-step span { color: var(--muted); font-size: 0.78rem; }
    .error { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.4); color: #fca5a5; padding: 1rem 1.25rem; border-radius: 10px; margin-bottom: 1.5rem; font-size: 0.9rem; }
    footer { border-top: 1px solid var(--border); padding: 2rem; text-align: center; color: var(--muted); font-size: 0.82rem; }
    footer b { color: var(--text); }
    /* Datasets page — single compact card */
    .datasets-page { min-height: calc(100vh - var(--nav-h) - 80px); display: flex; align-items: center; justify-content: center; padding: 3rem 1.5rem; }
    .download-card { max-width: 440px; width: 100%; }
    .customer-preview { display: flex; align-items: center; gap: 1rem; padding: 1rem; background: var(--surface2); border-radius: 12px; margin: 1.25rem 0; border: 1px solid var(--border); }
    .customer-preview .info b { display: block; font-size: 1rem; }
    .customer-preview .info span { color: var(--muted); font-size: 0.82rem; }
    .name-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 1.25rem; }
    .chip { font-size: 0.72rem; padding: 0.3rem 0.65rem; border-radius: 999px; background: rgba(59,130,246,0.08); color: var(--muted); border: 1px solid var(--border); cursor: pointer; transition: all 0.15s; }
    .chip:hover, .chip.active { background: rgba(59,130,246,0.2); color: #93c5fd; border-color: rgba(59,130,246,0.4); }
    .back-link { display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); text-decoration: none; font-size: 0.85rem; margin-bottom: 1.5rem; }
    .back-link:hover { color: var(--accent); }
"""

NAV = """
  <nav class="navbar">
    <a class="nav-brand" href="/">
      <div class="logo">🛡</div>FraudGuard
    </a>
    <ul class="nav-links">
      <li><a href="/" class="{{ 'active' if page == 'home' else '' }}">Home</a></li>
      <li><a href="/datasets" class="{{ 'active' if page == 'datasets' else '' }}">Download Dataset</a></li>
      <li><a href="/#test">Test Model</a></li>
      <li><a href="/#how">How It Works</a></li>
      <li><a href="/#test" class="nav-cta">Run Detection →</a></li>
    </ul>
  </nav>
"""

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FraudGuard — Credit Card Fraud Detection</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>""" + STYLES + """</style>
</head>
<body>
""" + NAV + """
  <div class="page">
    <section class="hero" id="home">
      <div class="container">
        <h1>Credit Card Fraud<br>Detection System</h1>
        <p>Upload a customer's transaction dataset and let our ML model detect fraudulent transactions in seconds.</p>
        <div class="hero-btns">
          <a class="btn btn-outline" href="/datasets">⬇ Download Dataset</a>
          <a class="btn btn-primary" href="#test">🔍 Test Model</a>
        </div>
        <div class="hero-stats">
          <div class="hero-stat"><b>10</b><span>Customer Profiles</span></div>
          <div class="hero-stat"><b>100</b><span>Transactions Each</span></div>
          <div class="hero-stat"><b>6</b><span>ML Models</span></div>
          <div class="hero-stat"><b>Group 4</b><span>Threat Detection</span></div>
        </div>
      </div>
    </section>

    <section id="test">
      <div class="container">
        <div class="section-label">Test Model</div>
        <h2 class="section-title">Upload &amp; Test the Model</h2>
        <p class="section-desc">Upload the CSV you downloaded and select an ML model to run fraud detection.</p>

        {% if error %}<div class="error">⚠ {{ error }}</div>{% endif %}

        <div class="test-grid">
          <div class="card">
            <form method="post" enctype="multipart/form-data">
              <div class="form-group">
                <label>Select ML Model</label>
                <select name="model">
                  {% for m in models %}
                  <option value="{{ m }}" {% if m == selected %}selected{% endif %}>{{ m }}</option>
                  {% endfor %}
                </select>
              </div>
              <div class="form-group">
                <label>Upload CSV File</label>
                <label class="file-drop" for="csvfile">
                  <div class="icon">📂</div>
                  <p>Click to browse or drag &amp; drop<br><small>100-transaction CSV file</small></p>
                  <input type="file" name="csvfile" id="csvfile" accept=".csv" required
                         onchange="document.getElementById('fname').textContent=this.files[0]?.name||''">
                </label>
                <div class="file-name-display" id="fname"></div>
              </div>
              <button type="submit" class="btn btn-primary" style="width:100%;justify-content:center">🚀 Run Fraud Detection</button>
            </form>
            <p style="color:var(--muted);font-size:0.78rem;margin-top:1rem">Need a dataset? <a href="/datasets" style="color:var(--accent)">Download here →</a></p>
          </div>
          <div class="card">
            <p style="font-weight:700;margin-bottom:1rem">How to test</p>
            <ul class="steps-list">
              <li><div class="step-num">1</div><div class="step-text"><b>Download a dataset</b><span>Go to Download Dataset page &amp; pick a customer</span></div></li>
              <li><div class="step-num">2</div><div class="step-text"><b>Upload the CSV</b><span>Use the exact file you downloaded</span></div></li>
              <li><div class="step-num">3</div><div class="step-text"><b>Choose a model &amp; run</b><span>Click Run Fraud Detection</span></div></li>
              <li><div class="step-num">4</div><div class="step-text"><b>View the report</b><span>See fraud cases detected for that customer</span></div></li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    {% if person and summary %}
    <section id="results" style="background:rgba(17,24,39,0.5)">
      <div class="container">
        <div class="section-label">Results</div>
        <h2 class="section-title">Fraud Detection Report</h2>
        <div class="result-hero">
          <div class="avatar">{{ person.name.split()[0][0] }}{{ person.name.split()[1][0] }}</div>
          <div>
            <h2>{{ person.name }}</h2>
            <p>📍 {{ person.city }} · 100 transactions · {{ selected }}</p>
          </div>
        </div>
        <div class="metrics">
          <div class="metric"><b>{{ summary.total_transactions }}</b><span>Total Transactions</span></div>
          <div class="metric danger"><b>{{ summary.flagged_fraud }}</b><span>Fraud Detected</span></div>
          <div class="metric success"><b>{{ summary.flagged_normal }}</b><span>Normal</span></div>
          <div class="metric danger"><b>{{ person.actual_fraud_cases }}</b><span>Actual Fraud</span></div>
          {% if summary.accuracy is defined %}<div class="metric success"><b>{{ '%.0f'|format(summary.accuracy * 100) }}%</b><span>Accuracy</span></div>{% endif %}
          {% if summary.fraud_recall is defined %}<div class="metric"><b>{{ '%.0f'|format(summary.fraud_recall * 100) }}%</b><span>Fraud Recall</span></div>{% endif %}
        </div>
        <div class="insight">✅ <b>{{ person.name }}</b> had <b>{{ person.actual_fraud_cases }} actual fraud</b> out of 100. Model flagged <b>{{ summary.flagged_fraud }}</b> as fraud.</div>
        <div class="card" style="margin-top:1.5rem">
          <details><summary style="cursor:pointer;color:var(--muted)">📋 View all 100 predictions</summary>
            <div class="table-wrap"><table>
              <tr><th>#</th><th>Prediction</th><th>Prob.</th><th>Actual</th><th>Match</th></tr>
              {% for r in rows %}
              <tr>
                <td>{{ r.row }}</td>
                <td class="{{ 'fraud' if r.prediction == 1 else 'normal' }}">{{ r.result_label }}</td>
                <td>{{ '%.1f'|format(r.fraud_probability * 100) }}%</td>
                <td class="{{ 'fraud' if r.actual_class == 1 else 'normal' }}">{{ r.actual_label }}</td>
                <td>{{ '✓' if r.correct else '✗' }}</td>
              </tr>
              {% endfor %}
            </table></div>
          </details>
        </div>
      </div>
    </section>
    {% endif %}

    <section id="how">
      <div class="container">
        <div class="section-label">About</div>
        <h2 class="section-title">How It Works</h2>
        <p class="section-desc">Our pipeline processes raw transaction data through multiple ML stages.</p>
        <div class="pipeline">
          <div class="pipe-step"><div class="icon">⚖️</div><b>RobustScaler</b><span>Scales Time &amp; Amount</span></div>
          <div class="pipe-step"><div class="icon">🎯</div><b>Feature Selection</b><span>15 V-features</span></div>
          <div class="pipe-step"><div class="icon">📉</div><b>PCA</b><span>6 components</span></div>
          <div class="pipe-step"><div class="icon">🤖</div><b>ML Model</b><span>Normal or Fraud</span></div>
        </div>
      </div>
    </section>
    <footer><b>FraudGuard</b> — Group 4 Threat Detection Project</footer>
  </div>
  {% if person and summary %}<script>document.getElementById('results')?.scrollIntoView({behavior:'smooth'});</script>{% endif %}
</body></html>
"""

DATASETS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Download Dataset — FraudGuard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>""" + STYLES + """</style>
</head>
<body>
""" + NAV + """
  <div class="page">
    <div class="datasets-page">
      <div class="download-card">
        <a class="back-link" href="/">← Back to Home</a>
        <div class="card">
          <div class="section-label">Download</div>
          <h2 class="section-title" style="font-size:1.35rem;margin-bottom:0.25rem">Transaction Dataset</h2>
          <p class="section-desc" style="margin-bottom:1.25rem;font-size:0.88rem">Select a customer to download their 100-transaction CSV file.</p>

          <div class="form-group">
            <label>Select Customer</label>
            <select id="personSelect" onchange="updatePreview()">
              {% for p in persons %}
              <option value="{{ p.id }}" data-name="{{ p.name }}" data-city="{{ p.city }}"
                      data-initials="{{ p.name.split()[0][0] }}{{ p.name.split()[1][0] }}">
                {{ p.name }} — {{ p.city }}
              </option>
              {% endfor %}
            </select>
          </div>

          <div class="customer-preview" id="preview">
            <div class="avatar" id="prevInitials">JM</div>
            <div class="info">
              <b id="prevName">James Mitchell</b>
              <span id="prevCity">📍 Chicago, IL · 100 transactions · CSV</span>
            </div>
          </div>

          <a class="btn btn-green" id="downloadBtn" href="/download/person/james_mitchell">⬇ Download Dataset</a>

          <div class="name-chips">
            {% for p in persons %}
            <span class="chip {{ 'active' if loop.first else '' }}" data-id="{{ p.id }}" onclick="selectPerson('{{ p.id }}')">{{ p.name.split()[0] }}</span>
            {% endfor %}
          </div>
        </div>
      </div>
    </div>
    <footer><b>FraudGuard</b> — Group 4 Threat Detection Project</footer>
  </div>
  <script>
    const sel = document.getElementById('personSelect');
    function updatePreview() {
      const opt = sel.options[sel.selectedIndex];
      document.getElementById('prevInitials').textContent = opt.dataset.initials;
      document.getElementById('prevName').textContent = opt.dataset.name;
      document.getElementById('prevCity').textContent = '📍 ' + opt.dataset.city + ' · 100 transactions · CSV';
      document.getElementById('downloadBtn').href = '/download/person/' + opt.value;
      document.querySelectorAll('.chip').forEach(c => c.classList.toggle('active', c.dataset.id === opt.value));
    }
    function selectPerson(id) {
      sel.value = id;
      updatePreview();
    }
  </script>
</body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    summary = None
    rows = None
    person = None
    selected = list(AVAILABLE_MODELS.keys())[0]

    if request.method == "POST":
        selected = request.form.get("model", selected)
        upload = request.files.get("csvfile")
        if not upload or not upload.filename:
            error = "Please choose a CSV file."
        else:
            try:
                df = pd.read_csv(upload)
                person = identify_person(df, upload.filename)
                if person:
                    labeled = get_labeled_dataset(person["id"])
                    if labeled is not None:
                        df = labeled
                if len(df) != 100:
                    error = f"Expected 100 transactions, got {len(df)}. Download a dataset from the Download Dataset page."
                else:
                    results = predict_fraud(df, selected)
                    summary = summarize_results(results)
                    rows = results.to_dict("records")
                    if person is None:
                        error = "Could not identify the customer. Upload the exact file you downloaded."
                        summary = rows = None
            except Exception as exc:
                error = str(exc)

    return render_template_string(
        HOME_HTML,
        page="home",
        models=list(AVAILABLE_MODELS.keys()),
        unavailable=", ".join(UNAVAILABLE_MODELS),
        selected=selected,
        error=error,
        summary=summary,
        rows=rows,
        person=person,
    )


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/datasets")
def datasets_page():
    return render_template_string(
        DATASETS_HTML,
        page="datasets",
        persons=load_persons(),
    )


@app.route("/download/person/<person_id>")
def download_person(person_id: str):
    persons = {p["id"]: p for p in load_persons()}
    person = persons.get(person_id)
    path = DATASET_DIR / f"{person_id}.csv"
    if not person or not path.exists():
        return "Dataset not found", 404
    return send_file(path, as_attachment=True, download_name=person["download_name"])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  FraudGuard Website starting...")
    print("  Open: http://127.0.0.1:5000")
    print("  Keep this window OPEN while using the site.")
    print("=" * 50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
