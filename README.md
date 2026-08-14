# Credit Card Fraud Detection Demo

ML deployment demo for Group 4 Threat Detection project.

## Quick start

```powershell
pip install -r requirements.txt
streamlit run app.py
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for free hosting on **Streamlit Cloud** or **Vercel**.

## Project structure

| File / folder | Purpose |
|---------------|---------|
| `app.py` | Streamlit web app |
| `predict.py` | Preprocessing + inference pipeline |
| `model/` | Trained joblib models + scaler/PCA |
| `sample_upload_data.csv` | Sample file for users to download |
| `vercel-app/` | Next.js + Python API for Vercel hosting |

## Pipeline

1. RobustScaler on Time & Amount
2. Feature selection (15 V-features)
3. PCA (6 components)
4. ML model prediction
