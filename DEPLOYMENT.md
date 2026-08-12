# FedMed AI — Complete Local Testing & Deployment Guide

This guide covers local development, model training, artifact generation, testing, and deployment to **Render** (for the Flask backend) and **GitHub Pages** (for the static Doctor Dashboard frontend).

---

## 1. Project Architecture Overview

```
┌──────────────────────────────────────┐                HTTP (fetch)               ┌─────────────────────────────────────┐
│  Frontend (GitHub Pages / Static)    │  ──────────────────────────────────────▶  │  Backend (Render Web Service)       │
│  • HTML5 + Vanilla JS + Chart.js     │                                           │  • Flask + Gunicorn + TensorFlow    │
│  • Config-based API URL (config.js)  │  ◀──────────────────────────────────────  │  • Loads saved .keras model         │
│  • Works locally & in cloud          │                JSON / Images              │  • Inference ONLY — NO dataset req. │
└──────────────────────────────────────┘                                           └─────────────────────────────────────┘
```

- **Separation of Concerns**: Training is completely decoupled from the production backend. The backend loads pre-trained model weights (`backend/models/fedrad_global_model.keras`) and JSON artifacts on startup.
- **Zero-Retraining on Render**: Deploying to Render requires NO dataset downloads and runs zero training epochs on startup.

---

## 2. Local Setup & Testing

### Step 2.1: Clone and Install Dependencies

```bash
# Navigate to the project root
cd "FedMed AI"

# Install backend production dependencies
pip install -r backend/requirements.txt

# (Optional) Install training dependencies if you intend to train locally
pip install -r training/requirements_training.txt
```

### Step 2.2: (Optional) Model Training & Artifact Generation

*Skip this step if you already have the trained `.keras` model in `backend/models/`.*

```bash
# 1. Provide Kaggle API credentials (download kaggle.json from Kaggle Account page)
# Place kaggle.json in the project root or ~/.kaggle/kaggle.json

# 2. Download and extract the dataset (~1-2 GB)
python training/download_dataset.py

# 3. Option A: Full Training (GPU recommended, ~10 rounds)
python training/train.py

# 3. Option B: Fast CPU Smoke-Test (~2 rounds, 112x112 images)
python training/train.py --fast
```

This generates all required artifacts in `backend/models/`:
- `fedrad_global_model.keras`
- `labels.json`
- `model_metadata.json`
- `model_evaluation.json`
- `roc_pr_curves.json`
- `confusion_matrix.json`
- `fl_history.json`
- `hospital_round_logs.json`
- `dataset_stats.json`

### Step 2.3: Run the Backend Locally

```bash
cd backend
python app.py
```
The server starts at `http://127.0.0.1:5000`.

### Step 2.4: Test Backend Endpoints Locally

Open a new terminal and run health check and prediction tests:

```bash
# Health Check
curl http://127.0.0.1:5000/health

# Dashboard Overview
curl http://127.0.0.1:5000/api/dashboard/overview

# Federated Learning Status
curl http://127.0.0.1:5000/api/fl/status

# Test Prediction API (replace sample.png with a real X-ray image path)
curl -X POST -F "image=@sample.png" -F 'patient={"name":"Test Patient","age":45}' http://127.0.0.1:5000/api/predict
```

### Step 2.5: Run the Frontend Locally

Simply open `frontend/index.html` or `frontend/login.html` in your browser, or serve it using Python:

```bash
cd frontend
python -m http.server 8000
```
Open `http://localhost:8000/login.html` in your browser.
Default credentials for login: `dr.mehta@stmarcus-hosp.org` / `demo1234`.

---

## 3. Render Web Service Deployment (Backend)

### Step 3.1: Large Model File Storage (Git LFS / Hugging Face)

The trained DenseNet121 model (`fedrad_global_model.keras`) is typically ~75–150 MB.

#### Option A: Git LFS (Recommended)
This repository contains a `.gitattributes` configured for Git LFS.

```bash
# Install Git LFS locally
git lfs install

# Track keras model files
git lfs track "*.keras"

# Commit and push
git add .gitattributes backend/models/fedrad_global_model.keras
git commit -m "Add pre-trained model with Git LFS"
git push origin main
```

#### Option B: Hugging Face / External Direct Link
If you prefer not to use Git LFS on GitHub, upload `fedrad_global_model.keras` to a Hugging Face model repository or GitHub Release, and modify `backend/app.py` or a download script to fetch it if missing.

### Step 3.2: Render Configuration

1. Log into [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository.
4. Fill out the configuration options:

| Setting | Value |
|---|---|
| **Name** | `fedmedai` |
| **Region** | Choose closest to you (e.g. Oregon / Frankfurt) |
| **Branch** | `main` |
| **Root Directory** | *(leave empty)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r backend/requirements.txt` |
| **Start Command** | `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120` |

### Step 3.3: Environment Variables on Render

Under **Advanced** → **Environment Variables**, add:

| Key | Value | Notes |
|---|---|---|
| `PYTHON_VERSION` | `3.10.12` | Python version |
| `MODEL_DIR` | `models` | Model folder |
| `GRADCAM_DIR` | `gradcam_outputs` | Grad-CAM output folder |
| `UPLOADS_DIR` | `uploads` | Temp upload folder |
| `CORS_ORIGINS` | `*` | Or set to your GitHub Pages URL |
| `FLASK_DEBUG` | `false` | Production flag |

5. Click **Create Web Service**.
6. Once deployed, Render will provide a URL, e.g. `https://fedmedai.onrender.com`.
7. Verify by navigating to `https://fedmedai.onrender.com/health` in your browser.

---

## 4. GitHub Pages Deployment (Frontend)

### Step 4.1: Update `config.js` with Render URL

In `frontend/js/config.js`, update `window.BACKEND_URL`:

```javascript
window.BACKEND_URL = window.BACKEND_URL || "https://fedmedai.onrender.com/api";
```

### Step 4.2: Deploy via GitHub Actions (Automated)

This repository includes `.github/workflows/pages.yml`.

1. Go to your GitHub repository → **Settings** → **Pages**.
2. Under **Build and deployment** → **Source**, select **GitHub Actions**.
3. Push changes to `main`:
   ```bash
   git add .
   git commit -m "Configure production backend URL and GitHub Pages deployment"
   git push origin main
   ```
4. The Action will automatically build and deploy `frontend/` to `https://<your-username>.github.io/<repo-name>/`.

---

## 5. Verification Checklist

- [x] Backend runs independently using `python backend/app.py` or `gunicorn backend.app:app`
- [x] Pre-trained model loads on startup without downloading dataset or running training epochs
- [x] `/health` endpoint returns status `ok` and 200 HTTP response
- [x] Uploading a chest X-ray image triggers prediction, returns top disease findings & confidence scores
- [x] Grad-CAM overlay image is generated and served correctly
- [x] Overview dashboard, disease stats, dataset stats, FL monitor, and model performance graphs all display real data without blank or broken charts
- [x] Frontend dynamically connects to backend using environment/config URL (no hardcoded localhost in JS)
- [x] Production backend ready for Render deployment (`render.yaml`, `requirements.txt`, `gunicorn`)
- [x] Frontend ready for GitHub Pages deployment (`404.html`, GitHub Actions workflow)
