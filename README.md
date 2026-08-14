# FraudGuard — Credit Card Fraud Detection Demo

Group 4 Threat Detection Project · ML deployment demo

## Run locally

```powershell
pip install -r requirements.txt
python flask_app.py
```

Or double-click **`run.bat`** → open http://127.0.0.1:5000

## Host online (free)

See **[DEPLOYMENT.md](DEPLOYMENT.md)** — recommended: **Render.com**

## Project structure

| File / folder | Purpose |
|---------------|---------|
| `flask_app.py` | Main web app (Flask) |
| `predict.py` | ML inference pipeline |
| `person_data/` | 10 customer datasets (100 tx each) |
| `model/` | Trained joblib models |
| `vercel-app/` | Vercel deployment (lite models) |

## Git push (if error)

```powershell
git pull origin main --allow-unrelated-histories
git add .
git commit -m "Your message"
git push origin main
```

Do **not** commit `creditcard.csv` (150 MB — too large for GitHub).
