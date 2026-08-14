# How to Host FraudGuard Online (Free)

Share a live link with your teacher. Three options below — **Render is recommended** for your full Flask app.

---

## Before you deploy (all platforms)

### 1. Create a GitHub account
https://github.com — free

### 2. Upload project to GitHub

Create a new repository (e.g. `fraud-detection-demo`), then upload **these files/folders**:

```
flask_app.py
predict.py
person_data.py
requirements.txt
Procfile
render.yaml          (for Render only)
model/               (entire folder)
person_data/         (entire folder)
generate_person_datasets.py
```

### 3. Do NOT upload these (too large or not needed)

```
creditcard.csv       (150 MB — too big)
.server.lock
run.bat / stop.bat / restart.bat
__pycache__/
.streamlit/
vercel-app/          (only if using Vercel option)
app.py               (Streamlit — not needed for Flask deploy)
```

---

## Option A: Render.com (RECOMMENDED)

Best for your Flask app + all ML models. **100% free** (with limits).

### Steps

1. Push your code to GitHub (see above)

2. Go to **https://render.com** → Sign up with GitHub (free)

3. Click **New +** → **Web Service**

4. Connect your GitHub repo

5. Settings:
   | Field | Value |
   |-------|-------|
   | Name | `fraudguard` (or any name) |
   | Region | Singapore (closest to Bangladesh) |
   | Branch | `main` |
   | Runtime | **Python 3** |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn flask_app:app --bind 0.0.0.0:$PORT` |
   | Plan | **Free** |

6. Click **Create Web Service**

7. Wait 5–10 minutes for first deploy

8. Your live URL will be:
   ```
   https://fraudguard.onrender.com
   ```
   (or whatever name you chose)

### Important notes for Render free tier
- App **sleeps after 15 min** of no use — first visit takes ~30 sec to wake up
- Tell your teacher: "Wait a moment on first load"
- Free tier has enough space for your models (~180 MB)

---

## Option B: Vercel.com

Good if teacher specifically asked for Vercel. **Limitation:** large models (Random Forest, KNN) won't fit — only 4 small models work.

### What to deploy
Use the `vercel-app/` folder (Next.js + Python API), NOT the main Flask app.

### Steps

1. Install Node.js: https://nodejs.org

2. Create GitHub repo with **only** the `vercel-app/` contents:
   ```
   vercel-app/
     api/predict.py
     app/page.tsx, layout.tsx, globals.css
     model-lite/        (small models only)
     public/            (sample CSVs)
     package.json
     requirements.txt
     vercel.json
     next.config.js
     tsconfig.json
   ```

3. Go to **https://vercel.com** → Sign up with GitHub

4. Click **Add New Project** → Import your repo

5. Settings:
   | Field | Value |
   |-------|-------|
   | Root Directory | `.` (or `vercel-app` if repo root is parent) |
   | Framework | Next.js (auto-detected) |

6. Click **Deploy**

7. Live URL:
   ```
   https://your-project.vercel.app
   ```

### Vercel CLI (alternative)
```powershell
cd C:\Users\USER\Downloads\saved_models\vercel-app
npm install
npx vercel login
npx vercel --prod
```

### Vercel limitations
- No 10-customer download page (old simpler UI)
- Only Logistic Regression, Decision Tree, MLP, SVM models
- Random Forest & KNN too large for Vercel limits

---

## Option C: Hugging Face Spaces (ML demo)

Good free ML hosting with GPU option later.

### Steps

1. Go to **https://huggingface.co** → Create account

2. Click **New Space** → Choose **Docker** or **Gradio**

3. Upload your Flask app files OR create a Gradio wrapper

4. For simplest approach, use **Gradio**:
   ```python
   import gradio as gr
   # wrap predict_fraud() function
   ```

5. Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/fraud-detection`

*(More setup required — use Render if you want the exact current UI)*

---

## Option D: PythonAnywhere (simple Flask)

Free tier: https://www.pythonanywhere.com

1. Sign up (free Beginner account)
2. Upload files via **Files** tab
3. Create new **Web App** → Manual config → Flask
4. Point WSGI file to `flask_app.py`
5. URL: `https://YOURUSERNAME.pythonanywhere.com`

**Limitation:** Free tier can't install heavy packages easily; Render is better.

---

## Comparison table

| Platform | Free? | Full UI? | All Models? | Best for |
|----------|-------|----------|-------------|----------|
| **Render** | ✅ | ✅ | ✅ | **Your project (best)** |
| Vercel | ✅ | Partial | ❌ (4 models) | Teacher asked Vercel |
| Hugging Face | ✅ | Custom | ✅ | ML showcase |
| PythonAnywhere | ✅ | ✅ | ⚠️ | Simple demos |

---

## Recommended: Render step-by-step (quick version)

```
1. github.com → New repo → Upload project files
2. render.com → Sign up with GitHub
3. New Web Service → Connect repo
4. Build: pip install -r requirements.txt
5. Start:  gunicorn flask_app:app --bind 0.0.0.0:$PORT
6. Plan: Free → Deploy
7. Share URL with teacher ✅
```

---

## After deployment — test checklist

- [ ] Home page loads with navbar
- [ ] Download Dataset page shows customer dropdown
- [ ] CSV download works for James Mitchell
- [ ] Upload CSV + Run Detection shows results
- [ ] Customer name and fraud count display correctly

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails on Render | Check `requirements.txt` has `gunicorn` |
| Model not found | Ensure `model/` folder is in GitHub repo |
| App sleeps (Render) | Normal on free tier — refresh after 30 sec |
| Vercel too large | Use `model-lite/` folder only |
| sklearn version error | Pin `scikit-learn==1.6.1` in requirements |

---

**Group 4 – Threat Detection Project**
