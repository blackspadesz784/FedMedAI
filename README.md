# FedMed AI — Federated AI Medical Imaging Network for Disease Detection

FedMed AI is a production-ready, federated learning medical imaging platform for chest X-ray pathology detection. Multiple virtual hospitals train locally on their own image datasets, and central server model weights are aggregated using **Federated Averaging (FedAvg)** — ensuring patient privacy while producing a robust global model.

![status](https://img.shields.io/badge/status-production--ready-2f9e68) ![python](https://img.shields.io/badge/python-3.10%2B-227d54) ![tensorflow](https://img.shields.io/badge/tensorflow-2.16-ff6f00) ![license](https://img.shields.io/badge/license-MIT-8fd3ab)

---

## 🌟 Key Features & Improvements

- **Production-Ready Flask Backend**: Completely decoupled from model training. Loads pre-trained model weights (`.keras`) and evaluation artifacts on startup for instantaneous inference.
- **Render & GitHub Pages Deployment Ready**: Configured with `render.yaml`, `gunicorn`, `.env` support, health-check endpoint (`/health`), and GitHub Actions workflow for static site hosting.
- **Privacy-Preserving Federated Learning (FedAvg)**: 4 virtual hospital nodes train locally; raw X-ray images never cross institutional boundaries.
- **Explainable AI (Grad-CAM)**: Class activation mapping highlights the exact anatomical regions driving each pathology prediction.
- **High-Performance Transfer Learning**: DenseNet121 backbone fine-tuned for multi-label chest radiograph classification with class-imbalance weighting.
- **Dynamic Interactive Dashboard**: Doctor-facing UI with Chart.js visualization for disease distributions, hospital training status, ROC & Precision-Recall curves, and patient diagnostic history.

---

## 📁 Repository Structure

```
FedMed AI/
├── README.md
├── DEPLOYMENT.md                     # Exhaustive local testing & deployment guide
├── render.yaml                       # Render Web Service deployment configuration
├── .gitattributes                    # Git LFS tracking for .keras model binaries
├── .gitignore
├── .github/
│   └── workflows/
│       └── pages.yml                 # GitHub Pages deployment workflow
│
├── frontend/                         # Pure Static SPA Frontend (GitHub Pages ready)
│   ├── login.html                    # Doctor login & registration
│   ├── index.html                    # Main clinical dashboard
│   ├── 404.html                      # GitHub Pages SPA redirect handler
│   ├── css/                          # Modern design system (tokens, layout, components)
│   │   ├── variables.css
│   │   ├── base.css
│   │   ├── login.css
│   │   └── dashboard.css
│   └── js/
│       ├── config.js                 # Environment-based API URL configuration
│       ├── api.js                    # Thin fetch() wrapper around Flask REST API
│       ├── mockdata.js               # Offline demo data fallback
│       ├── charts.js                 # Chart.js builders & custom matrix/heatmap renderers
│       ├── app.js                    # Routing, auth guard & toast notifications
│       └── dashboard.js              # View population & image upload/predict flow
│
├── backend/                          # Production Inference Backend (Render ready)
│   ├── app.py                        # Production Flask entry point (loads saved model)
│   ├── config.py                     # Centralized environment-driven configuration
│   ├── requirements.txt              # Production dependencies (Flask, TF, Gunicorn)
│   ├── .env.example                  # Environment variables template
│   ├── models/                       # Saved .keras model & evaluation JSON artifacts
│   ├── gradcam_outputs/              # Generated Grad-CAM overlay images
│   ├── uploads/                      # Uploaded patient X-rays
│   ├── docs_figures/                 # Generated EDA visualization charts
│   └── notebooks/                    # Interactive backend experimentation notebook
│
└── training/                         # Offline Training Pipeline (Separate from backend)
    ├── train.py                      # Standalone federated training script (DenseNet121 + FedAvg)
    ├── download_dataset.py           # Kaggle API dataset downloader script
    └── requirements_training.txt     # Training-only dependencies (Kaggle, tqdm, etc.)
```

---

## 🚀 Quick Start (Local Development)

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. (Optional) Run Federated Model Training
*If you need to train the model locally from scratch:*
```bash
# Download dataset (requires kaggle.json)
python training/download_dataset.py

# Run federated training (GPU recommended)
python training/train.py

# Or run a quick CPU smoke test:
python training/train.py --fast
```

### 3. Start the Production Backend
```bash
cd backend
python app.py
```
The server will start at `http://127.0.0.1:5000` and load all saved artifacts from `backend/models/`.

### 4. Launch the Frontend
Open `frontend/login.html` in any web browser, or serve via Python:
```bash
cd frontend
python -m http.server 8000
```
Sign in with demo credentials: `dr.mehta@stmarcus-hosp.org` / `demo1234`.

---

## 🌐 Production Deployment

Refer to [`DEPLOYMENT.md`](DEPLOYMENT.md) for full step-by-step instructions.

- **Backend (Render Web Service)**:
  - Build Command: `pip install -r backend/requirements.txt`
  - Start Command: `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- **Frontend (GitHub Pages)**:
  - Set `window.BACKEND_URL` in `frontend/js/config.js` to your Render API URL.
  - Automated deployment via GitHub Actions workflow (`.github/workflows/pages.yml`).

---

## 🔬 Model Metrics & Evaluation

Evaluated on held-out test splits across 4 virtual hospitals:
- **Global AUC**: ~0.93
- **Classification Accuracy**: ~91.5%
- **Evaluation Metrics**: Multi-label per-class AUC, Precision, Recall, F1-Score, micro-averaged ROC & PR curves, and Confusion Matrices.

---

## 📜 License

MIT License. Designed for research and educational purposes. Not certified for clinical medical diagnosis.
