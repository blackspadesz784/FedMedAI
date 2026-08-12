"""
config.py — FedMed AI Backend Configuration
Reads all settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path

# ── Base paths ────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent
MODEL_DIR = BACKEND_DIR / os.getenv("MODEL_DIR", "models")
GRADCAM_DIR = BACKEND_DIR / os.getenv("GRADCAM_DIR", "gradcam_outputs")
UPLOADS_DIR = BACKEND_DIR / os.getenv("UPLOADS_DIR", "uploads")
FIGURES_DIR = BACKEND_DIR / "docs_figures"

# ── Model files ───────────────────────────────────────────────────────────────
GLOBAL_MODEL_PATH = MODEL_DIR / "fedmed_global_model.keras"
SKLEARN_MODEL_PATH = MODEL_DIR / "fedmed_global_model.joblib"
LABELS_PATH = MODEL_DIR / "labels.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_EVALUATION_PATH = MODEL_DIR / "model_evaluation.json"
ROC_PR_CURVES_PATH = MODEL_DIR / "roc_pr_curves.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.json"
FL_HISTORY_PATH = MODEL_DIR / "fl_history.json"
HOSPITAL_ROUND_LOGS_PATH = MODEL_DIR / "hospital_round_logs.json"
DATASET_STATS_PATH = MODEL_DIR / "dataset_stats.json"

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ── Security ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "fedmed-ai-dev-secret-change-in-production")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 16))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins; "*" allows all (fine for public API).
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# ── Inference ─────────────────────────────────────────────────────────────────
IMG_SIZE = int(os.getenv("IMG_SIZE", 64))
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", 0.5))
NUM_HOSPITALS = 4
HOSPITAL_NAMES = ["Hospital A", "Hospital B", "Hospital C", "Hospital D"]

# Ensure runtime dirs exist
for d in [MODEL_DIR, GRADCAM_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
