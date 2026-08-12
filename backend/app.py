"""
app.py — FedMed AI Production Flask Backend
============================================
Standalone production entry point. On startup this file:
  1. Loads the pre-trained global model (backend/models/fedrad_global_model.keras)
  2. Loads all JSON evaluation artifacts produced by training/train.py
  3. Starts the Flask REST API

IMPORTANT: This file contains ZERO training code. It never downloads a dataset
or re-trains the model. Training is a completely separate concern handled by
training/train.py which is run once locally before deployment.

Local development:
    cd backend
    python app.py

Production (Render / Gunicorn):
    cd backend
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
"""

import base64
import datetime
import io
import json
import os
import sys
import uuid
from pathlib import Path

# ── Load env vars from .env (development only — Render injects them directly) ─
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not required in production

import config  # noqa: E402 — config reads env vars; import after dotenv

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, make_response
from werkzeug.utils import secure_filename

# Sklearn for inference (Python 3.14 compatible)
import joblib as _joblib

# Try TensorFlow for Grad-CAM; graceful fallback if unavailable
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    tf = None
    keras = None
    layers = None
    HAS_TF = False

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

# ── CORS: pure manual approach — guaranteed to work with GitHub Pages ─────────
# We do NOT use flask-cors to avoid duplicate-header conflicts.
# Every response gets the headers via after_request; OPTIONS preflight
# is handled by a dedicated catch-all route registered before all others.

@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = \
        "Content-Type, Authorization, Accept, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = \
        "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def _preflight(path):
    """Handle all CORS pre-flight OPTIONS requests."""
    resp = make_response("", 204)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = \
        "Content-Type, Authorization, Accept, X-Requested-With"
    resp.headers["Access-Control-Allow-Methods"] = \
        "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Global model state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
_model = None          # sklearn MultiOutputClassifier (joblib)
_mlb = None            # MultiLabelBinarizer fitted on training data
_labels = []
_img_size = 64         # must match training
_feature_size = None
_model_metadata = {}
_model_evaluation = {}
_roc_pr_curves = {}
_confusion_matrix = {}
_fl_history = {"round": [], "global_val_accuracy": [], "global_val_auc": [], "global_val_loss": []}
_hospital_round_logs = {}
_dataset_stats = {}
_model_load_error = None


def _load_json(path: Path, default=None):
    """Safely load a JSON file, return default if missing."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return default if default is not None else {}


def load_model_and_artifacts():
    """Called once at server startup. Loads sklearn model + all JSON artifacts."""
    global _model, _mlb, _labels, _img_size, _feature_size
    global _model_metadata, _model_evaluation
    global _roc_pr_curves, _confusion_matrix, _fl_history
    global _hospital_round_logs, _dataset_stats, _model_load_error

    print("\n" + "=" * 60)
    print("  FedMed AI Backend — Loading model & artifacts")
    print("=" * 60)

    # ── Load labels ───────────────────────────────────────────────────────────
    if not config.LABELS_PATH.exists():
        _model_load_error = (
            f"labels.json not found at {config.LABELS_PATH}. "
            "Run `python training/train_sklearn.py` first."
        )
        print(f"  ERROR: {_model_load_error}")
        # load artifacts even without model for dashboard data
    else:
        _labels = _load_json(config.LABELS_PATH, [])
        print(f"  [OK] Labels loaded: {len(_labels)} classes")

    # ── Load sklearn model (joblib) ───────────────────────────────────────────
    if config.SKLEARN_MODEL_PATH.exists():
        print(f"  Loading sklearn model from {config.SKLEARN_MODEL_PATH} ...")
        try:
            bundle = _joblib.load(str(config.SKLEARN_MODEL_PATH))
            _model = bundle["model"]
            _mlb = bundle.get("mlb")
            if not _labels:
                _labels = bundle.get("all_labels", [])
            _img_size = bundle.get("img_size", 64)
            _feature_size = bundle.get("feature_size")
            _model_load_error = None
            print(f"  [OK] Sklearn model loaded — {len(_labels)} classes, img_size={_img_size}")
        except Exception as e:
            _model_load_error = f"Failed to load sklearn model: {e}"
            print(f"  ERROR: {_model_load_error}")
    else:
        _model_load_error = (
            f"Model not found at {config.SKLEARN_MODEL_PATH}. "
            "Run `python training/train_sklearn.py` first."
        )
        print(f"  ERROR: {_model_load_error}")

    # ── Load JSON artifacts ───────────────────────────────────────────────────
    _model_metadata = _load_json(config.MODEL_METADATA_PATH, {
        "model_version": "1.0-fedavg-sklearn",
        "backbone": "LogisticRegression+HOG",
        "num_classes": len(_labels),
        "fl_rounds": 5,
        "aggregation": "FedAvg",
    })
    _model_evaluation = _load_json(config.MODEL_EVALUATION_PATH, {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
        "f1": 0.0, "auc": 0.0, "loss": 0.0,
    })
    _roc_pr_curves = _load_json(config.ROC_PR_CURVES_PATH, {
        "roc": {"fpr": [0, 1], "tpr": [0, 1]},
        "pr": {"recall": [0, 1], "precision": [1, 0]},
    })
    _confusion_matrix = _load_json(config.CONFUSION_MATRIX_PATH, {"labels": [], "matrix": []})
    _fl_history = _load_json(config.FL_HISTORY_PATH, {
        "round": [], "global_val_accuracy": [], "global_val_auc": [], "global_val_loss": []
    })
    _hospital_round_logs = _load_json(config.HOSPITAL_ROUND_LOGS_PATH, {})
    _dataset_stats = _load_json(config.DATASET_STATS_PATH, {
        "total_images": 0, "train": 0, "val": 0, "test": 0, "classes": len(_labels),
        "labels": _labels, "label_counts": {},
    })

    # ── Load prediction history ───────────────────────────────────────────────
    global PREDICTION_LOG
    if config.PREDICTIONS_PATH.exists():
        PREDICTION_LOG = _load_json(config.PREDICTIONS_PATH, [])
        print(f"  [OK] Loaded {len(PREDICTION_LOG)} historical predictions from {config.PREDICTIONS_PATH.name}")

    print(f"  [OK] All artifacts loaded.")
    print(f"  [OK] Backend ready — serving at http://{config.HOST}:{config.PORT}")
    print("=" * 60 + "\n")



# ─────────────────────────────────────────────────────────────────────────────
# Image feature extraction (mirrors training/train_sklearn.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_from_bytes(image_bytes: bytes, img_size: int = None) -> np.ndarray:
    """Extract HOG-like features from raw image bytes. Must match training pipeline."""
    img_size = img_size or _img_size or 64
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPEG, PNG, or BMP file.")
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0

    flat = img.flatten()
    stats = np.array([
        img.mean(), img.std(), img.min(), img.max(),
        np.percentile(img, 25), np.percentile(img, 75),
        np.percentile(img, 10), np.percentile(img, 90),
    ])
    hist, _ = np.histogram(img, bins=16, range=(0, 1))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    lap = cv2.Laplacian(img, cv2.CV_32F)
    lap_stats = np.array([lap.mean(), lap.std()])
    h2, w2 = img_size // 2, img_size // 2
    quads = [img[:h2, :w2], img[:h2, w2:], img[h2:, :w2], img[h2:, w2:]]
    quad_feats = np.array([[q.mean(), q.std()] for q in quads]).flatten()

    return np.concatenate([flat, stats, hist, lap_stats, quad_feats]).reshape(1, -1)


def image_bytes_to_bgr(image_bytes: bytes, img_size: int = 256) -> np.ndarray:
    """Decode image bytes to BGR numpy array (for saliency overlay display)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return cv2.resize(img, (img_size, img_size))


# ─────────────────────────────────────────────────────────────────────────────
# Saliency map (sklearn substitute for Grad-CAM)
# ─────────────────────────────────────────────────────────────────────────────
def make_saliency_heatmap(image_bytes: bytes, pred_index: int, img_size: int = None) -> np.ndarray:
    """
    Approximate saliency: perturb local image patches and measure prediction
    change for the target class. This is model-agnostic (works with any sklearn model).
    Returns a normalized heatmap of shape (img_size, img_size).
    """
    img_size = img_size or _img_size or 64
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return np.zeros((img_size, img_size), dtype=np.float32)
    img_gray = cv2.resize(img_gray, (img_size, img_size)).astype(np.float32) / 255.0

    heatmap = np.zeros((img_size, img_size), dtype=np.float32)
    patch_size = max(8, img_size // 8)
    step = patch_size // 2

    # Baseline prediction on unperturbed image
    base_feat = extract_features_from_bytes(image_bytes, img_size)
    base_proba = _model.estimators_[pred_index].predict_proba(base_feat)[0][1]

    for y in range(0, img_size - patch_size + 1, step):
        for x in range(0, img_size - patch_size + 1, step):
            perturbed = img_gray.copy()
            perturbed[y:y + patch_size, x:x + patch_size] = img_gray.mean()
            # Re-extract features from perturbed image
            flat = perturbed.flatten()
            stats = np.array([
                perturbed.mean(), perturbed.std(), perturbed.min(), perturbed.max(),
                np.percentile(perturbed, 25), np.percentile(perturbed, 75),
                np.percentile(perturbed, 10), np.percentile(perturbed, 90),
            ])
            hist_p, _ = np.histogram(perturbed, bins=16, range=(0, 1))
            hist_p = hist_p.astype(np.float32) / (hist_p.sum() + 1e-8)
            lap = cv2.Laplacian(perturbed, cv2.CV_32F)
            lap_stats = np.array([lap.mean(), lap.std()])
            h2, w2 = img_size // 2, img_size // 2
            quads = [perturbed[:h2, :w2], perturbed[:h2, w2:], perturbed[h2:, :w2], perturbed[h2:, w2:]]
            quad_feats = np.array([[q.mean(), q.std()] for q in quads]).flatten()
            feat = np.concatenate([flat, stats, hist_p, lap_stats, quad_feats]).reshape(1, -1)
            try:
                pert_proba = _model.estimators_[pred_index].predict_proba(feat)[0][1]
                sensitivity = base_proba - pert_proba
            except Exception:
                sensitivity = 0.0
            heatmap[y:y + patch_size, x:x + patch_size] += sensitivity

    # Normalize to [0, 1]
    hm_min, hm_max = heatmap.min(), heatmap.max()
    if hm_max > hm_min:
        heatmap = (heatmap - hm_min) / (hm_max - hm_min)
    return np.clip(heatmap, 0, 1)


def create_gradcam_overlay(orig_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> bytes:
    """Blend heatmap onto original image, return PNG bytes."""
    h, w = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, orig_bgr, 1 - alpha, 0)
    _, buf = cv2.imencode(".png", overlay)
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────────────────────
def get_top_diseases(scores: np.ndarray, labels: list, threshold: float = None) -> tuple:
    """Return (top_label, top_score, top_diseases_list) sorted by score descending."""
    threshold = threshold or config.PREDICTION_THRESHOLD
    ranked = sorted(zip(labels, scores.tolist()), key=lambda x: -x[1])
    top_label, top_score = ranked[0]
    top_diseases = [{"name": n, "score": round(float(s), 4)} for n, s in ranked[:5]]
    return top_label, round(float(top_score), 4), top_diseases


# ─────────────────────────────────────────────────────────────────────────────
# In-memory stores (swap for a database in a full production system)
# ─────────────────────────────────────────────────────────────────────────────
REGISTERED_DOCTORS = {
    "dr.mehta@stmarcus-hosp.org": {"name": "Dr. Aanya Mehta", "password": "demo1234"},
    "demo@fedmed.ai": {"name": "Demo Doctor", "password": "demo1234"},
}
PREDICTION_LOG = []  # loaded from disk on startup; new entries appended
GRADCAM_STORE = {}   # pred_id → PNG bytes

# ── 20 seed patient records (shown until real predictions are made) ───────────
SEED_PATIENTS = [
    {"id": "s0",  "patient": "Arjun Sharma",    "age": 54, "finding": "Pneumonia",         "confidence": 0.91, "hospital": "Hospital A", "date": "2026-08-12", "time": "09:41 AM", "gradcam_url": None},
    {"id": "s1",  "patient": "Priya Nair",      "age": 38, "finding": "Effusion",          "confidence": 0.86, "hospital": "Hospital B", "date": "2026-08-12", "time": "10:05 AM", "gradcam_url": None},
    {"id": "s2",  "patient": "Ramesh Gupta",    "age": 67, "finding": "Cardiomegaly",      "confidence": 0.79, "hospital": "Hospital C", "date": "2026-08-12", "time": "10:22 AM", "gradcam_url": None},
    {"id": "s3",  "patient": "Sunita Patel",    "age": 45, "finding": "Atelectasis",       "confidence": 0.83, "hospital": "Hospital D", "date": "2026-08-12", "time": "10:47 AM", "gradcam_url": None},
    {"id": "s4",  "patient": "Vikram Reddy",    "age": 59, "finding": "No Finding",        "confidence": 0.95, "hospital": "Hospital A", "date": "2026-08-11", "time": "08:30 AM", "gradcam_url": None},
    {"id": "s5",  "patient": "Deepa Menon",     "age": 42, "finding": "Infiltration",      "confidence": 0.88, "hospital": "Hospital B", "date": "2026-08-11", "time": "09:15 AM", "gradcam_url": None},
    {"id": "s6",  "patient": "Karan Singh",     "age": 71, "finding": "Pneumothorax",      "confidence": 0.76, "hospital": "Hospital C", "date": "2026-08-11", "time": "11:02 AM", "gradcam_url": None},
    {"id": "s7",  "patient": "Meera Iyer",      "age": 33, "finding": "No Finding",        "confidence": 0.97, "hospital": "Hospital D", "date": "2026-08-11", "time": "02:18 PM", "gradcam_url": None},
    {"id": "s8",  "patient": "Ananya Das",      "age": 50, "finding": "Edema",             "confidence": 0.81, "hospital": "Hospital A", "date": "2026-08-10", "time": "08:55 AM", "gradcam_url": None},
    {"id": "s9",  "patient": "Rohit Verma",     "age": 62, "finding": "Consolidation",     "confidence": 0.84, "hospital": "Hospital B", "date": "2026-08-10", "time": "09:40 AM", "gradcam_url": None},
    {"id": "s10", "patient": "Fatima Khan",     "age": 48, "finding": "Mass",              "confidence": 0.72, "hospital": "Hospital C", "date": "2026-08-10", "time": "11:30 AM", "gradcam_url": None},
    {"id": "s11", "patient": "Siddharth Rao",   "age": 55, "finding": "Pleural_Thickening","confidence": 0.78, "hospital": "Hospital D", "date": "2026-08-10", "time": "01:20 PM", "gradcam_url": None},
    {"id": "s12", "patient": "Pooja Mishra",    "age": 29, "finding": "No Finding",        "confidence": 0.93, "hospital": "Hospital A", "date": "2026-08-09", "time": "10:10 AM", "gradcam_url": None},
    {"id": "s13", "patient": "Aditya Kumar",    "age": 73, "finding": "Emphysema",         "confidence": 0.80, "hospital": "Hospital B", "date": "2026-08-09", "time": "11:45 AM", "gradcam_url": None},
    {"id": "s14", "patient": "Nisha Chopra",    "age": 41, "finding": "Atelectasis",       "confidence": 0.87, "hospital": "Hospital C", "date": "2026-08-09", "time": "02:00 PM", "gradcam_url": None},
    {"id": "s15", "patient": "Manish Joshi",    "age": 66, "finding": "Effusion",          "confidence": 0.89, "hospital": "Hospital D", "date": "2026-08-08", "time": "09:05 AM", "gradcam_url": None},
    {"id": "s16", "patient": "Kavita Pillai",   "age": 37, "finding": "Fibrosis",          "confidence": 0.74, "hospital": "Hospital A", "date": "2026-08-08", "time": "10:35 AM", "gradcam_url": None},
    {"id": "s17", "patient": "Suresh Bhat",     "age": 58, "finding": "Cardiomegaly",      "confidence": 0.82, "hospital": "Hospital B", "date": "2026-08-08", "time": "12:00 PM", "gradcam_url": None},
    {"id": "s18", "patient": "Anjali Tiwari",   "age": 44, "finding": "Infiltration",      "confidence": 0.85, "hospital": "Hospital C", "date": "2026-08-07", "time": "08:20 AM", "gradcam_url": None},
    {"id": "s19", "patient": "Rajesh Nambiar",  "age": 69, "finding": "Nodule",            "confidence": 0.77, "hospital": "Hospital D", "date": "2026-08-07", "time": "03:15 PM", "gradcam_url": None},
]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Health
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Render health-check endpoint. Returns 200 when model is loaded, 503 otherwise."""
    if _model_load_error:
        return jsonify({"status": "error", "message": _model_load_error}), 503
    if _model is None:
        return jsonify({"status": "loading", "message": "Model not yet loaded"}), 503
    return jsonify({
        "status": "ok",
        "model": _model_metadata.get("model_version", "1.0-fedavg-sklearn"),
        "classes": len(_labels),
        "labels": _labels,
    })


@app.route("/api/reload", methods=["POST", "GET"])
def reload_artifacts():
    """Hot-reload all JSON artifacts from disk without restarting the server."""
    global _model_metadata, _model_evaluation, _roc_pr_curves, _confusion_matrix
    global _fl_history, _hospital_round_logs, _dataset_stats, _labels

    _labels = _load_json(config.LABELS_PATH, _labels)
    _model_metadata = _load_json(config.MODEL_METADATA_PATH, _model_metadata)
    _model_evaluation = _load_json(config.MODEL_EVALUATION_PATH, _model_evaluation)
    _roc_pr_curves = _load_json(config.ROC_PR_CURVES_PATH, _roc_pr_curves)
    _confusion_matrix = _load_json(config.CONFUSION_MATRIX_PATH, _confusion_matrix)
    _fl_history = _load_json(config.FL_HISTORY_PATH, _fl_history)
    _hospital_round_logs = _load_json(config.HOSPITAL_ROUND_LOGS_PATH, _hospital_round_logs)
    _dataset_stats = _load_json(config.DATASET_STATS_PATH, _dataset_stats)
    print("  [OK] Artifacts hot-reloaded from disk.")
    return jsonify({
        "status": "ok",
        "total_images": _dataset_stats.get("total_images", 0),
        "labels": _labels,
        "fl_rounds": len(_fl_history.get("round", [])),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Auth
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def login():
    body = request.get_json(force=True, silent=True) or {}
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    user = REGISTERED_DOCTORS.get(email)
    if user and user["password"] == password:
        return jsonify({"doctor": {"name": user["name"], "email": email}})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/auth/register", methods=["POST"])
def register():
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("name", "").strip()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    if not (name and email and password):
        return jsonify({"error": "name, email and password are required"}), 400
    REGISTERED_DOCTORS[email] = {"name": name, "password": password}
    return jsonify({"doctor": {"name": name, "email": email}})


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Prediction + Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if _model is None:
        msg = _model_load_error or "Model not loaded. Run training/train_sklearn.py first."
        return jsonify({"error": msg}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send file as multipart field 'image'."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file received."}), 400
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Supported: {', '.join(config.ALLOWED_EXTENSIONS)}"
        }), 400

    patient_meta = {}
    try:
        patient_meta = json.loads(request.form.get("patient", "{}"))
    except json.JSONDecodeError:
        pass

    try:
        image_bytes = file.read()
        features = extract_features_from_bytes(image_bytes, _img_size)
    except Exception as e:
        return jsonify({"error": f"Image preprocessing failed: {e}"}), 400

    # -- Run sklearn inference --
    try:
        scores = np.array([
            est.predict_proba(features)[0][1]
            for est in _model.estimators_
        ])
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    top_label, top_score, top_diseases = get_top_diseases(scores, _labels)
    pred_index = _labels.index(top_label)

    # -- Saliency map (sklearn substitute for Grad-CAM) --
    gradcam_url = None
    pred_id = len(PREDICTION_LOG)
    try:
        heatmap = make_saliency_heatmap(image_bytes, pred_index, _img_size)
        orig_bgr = image_bytes_to_bgr(image_bytes, 256)
        gradcam_png = create_gradcam_overlay(orig_bgr, heatmap)
        GRADCAM_STORE[pred_id] = gradcam_png
        gradcam_url = f"/api/gradcam/{pred_id}"
    except Exception as e:
        print(f"  Saliency map error (non-fatal): {e}")

    # -- Log & persist prediction --
    entry = {
        "id": pred_id,
        "patient": patient_meta.get("name", "Unnamed patient"),
        "age": patient_meta.get("age", "—"),
        "finding": top_label,
        "confidence": top_score,
        "hospital": patient_meta.get("hospital", "Hospital A"),
        "date": datetime.date.today().isoformat(),
        "time": datetime.datetime.now().strftime("%I:%M %p"),
        "gradcam_url": gradcam_url,
    }
    PREDICTION_LOG.append(entry)

    # Save updated prediction log to disk
    try:
        with open(config.PREDICTIONS_PATH, "w") as f:
            json.dump(PREDICTION_LOG, f, indent=2)
    except Exception as e:
        print(f"  WARNING: Could not write {config.PREDICTIONS_PATH}: {e}")

    # Save Grad-CAM image to disk if generated
    if gradcam_url and pred_id in GRADCAM_STORE:
        try:
            g_path = config.GRADCAM_DIR / f"gradcam_{pred_id}.png"
            with open(g_path, "wb") as f:
                f.write(GRADCAM_STORE[pred_id])
        except Exception as e:
            print(f"  WARNING: Could not save Grad-CAM image: {e}")

    return jsonify({
        "top_disease": top_label,
        "confidence": top_score,
        "model_version": _model_metadata.get("model_version", "1.0-fedavg-sklearn"),
        "top_diseases": top_diseases,
        "gradcam_overlay_url": gradcam_url,
        "prediction_id": pred_id,
        "all_scores": [
            {"name": lbl, "score": round(float(s), 4)}
            for lbl, s in zip(_labels, scores.tolist())
        ],
    })


@app.route("/api/gradcam/<int:pred_id>", methods=["GET"])
def get_gradcam(pred_id):
    png_bytes = GRADCAM_STORE.get(pred_id)
    if png_bytes is None:
        # Check disk fallback
        disk_path = config.GRADCAM_DIR / f"gradcam_{pred_id}.png"
        if disk_path.exists():
            return send_file(str(disk_path), mimetype="image/png")
        return jsonify({"error": "Grad-CAM not found for this prediction ID."}), 404
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"gradcam_{pred_id}.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Patient history
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/patients/history", methods=["GET"])
def patient_history():
    q = request.args.get("q", "").lower()
    # Merge seed records + real predictions (real ones appear first / most recent)
    all_items = list(PREDICTION_LOG) + SEED_PATIENTS
    if q:
        all_items = [p for p in all_items if q in p.get("patient", "").lower()]
    return jsonify({"items": list(reversed(all_items))})


@app.route("/api/patients/<int:pred_id>", methods=["GET"])
def get_patient(pred_id):
    entry = next((p for p in PREDICTION_LOG if p["id"] == pred_id), None)
    if not entry:
        return jsonify({"error": "Patient record not found."}), 404
    return jsonify(entry)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Dashboard overview
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/dashboard/overview", methods=["GET"])
def dashboard_overview():
    today = datetime.date.today().isoformat()
    # Count real + seed patients for total
    all_records = list(PREDICTION_LOG) + SEED_PATIENTS
    predictions_today = sum(1 for p in PREDICTION_LOG if p.get("date") == today)
    # avg confidence across all records
    confidences = [p["confidence"] for p in all_records if "confidence" in p]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    return jsonify({
        "total_patients": len(all_records),
        "predictions_today": predictions_today,
        "avg_confidence": round(avg_confidence, 4),
        "active_hospitals": config.NUM_HOSPITALS,
        "model_version": _model_metadata.get("model_version", "1.0-fedavg"),
        "global_auc": round(float(_model_evaluation.get("auc", 0.0)), 4),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Disease + dataset statistics
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/dashboard/disease-stats", methods=["GET"])
def disease_stats():
    label_counts = _dataset_stats.get("label_counts", {})
    if not label_counts:
        return jsonify({"labels": _labels, "counts": [0] * len(_labels)})
    sorted_items = sorted(label_counts.items(), key=lambda x: -x[1])
    return jsonify({
        "labels": [k for k, _ in sorted_items],
        "counts": [v for _, v in sorted_items],
    })


@app.route("/api/dashboard/dataset-stats", methods=["GET"])
def dataset_stats():
    return jsonify({
        "total_images": _dataset_stats.get("total_images", 0),
        "train": _dataset_stats.get("train", 0),
        "val": _dataset_stats.get("val", 0),
        "test": _dataset_stats.get("test", 0),
        "classes": _dataset_stats.get("classes", len(_labels)),
        "labels": _labels,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Federated learning monitoring
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/fl/status", methods=["GET"])
def fl_status():
    rounds = _fl_history.get("round", [])
    aucs = _fl_history.get("global_val_auc", [])
    return jsonify({
        "current_round": rounds[-1] if rounds else 0,
        "total_rounds": _model_metadata.get("fl_rounds", len(rounds)),
        "global_auc": round(aucs[-1], 4) if aucs else 0.0,
        "status": "complete",
        "aggregation": "FedAvg",
        "num_hospitals": config.NUM_HOSPITALS,
    })


@app.route("/api/fl/hospitals", methods=["GET"])
def fl_hospitals():
    items = []
    local_epochs = _model_metadata.get("local_epochs", 3)
    for h in config.HOSPITAL_NAMES:
        logs = _hospital_round_logs.get(h, [])
        last = logs[-1] if logs else {}
        items.append({
            "name": h,
            "samples": int(last.get("samples", 0)),
            "epoch": f"{local_epochs}/{local_epochs}",
            "loss": round(float(last.get("val_loss", 0.0)), 4),
            "accuracy": round(float(last.get("val_accuracy", 0.0)), 4),
            "auc": round(float(last.get("val_auc", 0.0)), 4),
            "status": "Synced",
        })
    return jsonify({"items": items})


@app.route("/api/fl/training-curves", methods=["GET"])
def fl_training_curves():
    rounds = _fl_history.get("round", [])
    return jsonify({
        "labels": [f"R{r}" for r in rounds],
        "accuracy": _fl_history.get("global_val_accuracy", []),
        "auc": _fl_history.get("global_val_auc", []),
        "loss": _fl_history.get("global_val_loss", []),
    })


@app.route("/api/fl/evaluation", methods=["GET"])
def fl_evaluation():
    """Full evaluation endpoint — merges scalar metrics, curves, confusion matrix."""
    ev = dict(_model_evaluation)
    ev.pop("per_class", None)  # exclude from this response to keep payload small

    # accLossCurve — use FL rounds data for the "training curve" view
    rounds = _fl_history.get("round", [])
    acc_curve = _fl_history.get("global_val_accuracy", [])
    loss_curve = _fl_history.get("global_val_loss", [])

    ev["accLossCurve"] = {
        "labels": [f"Round {r}" for r in rounds],
        "train_acc": acc_curve,
        "val_acc": acc_curve,      # In FL, global val is the only tracked curve
        "train_loss": loss_curve,
        "val_loss": loss_curve,
    }
    ev["roc"] = _roc_pr_curves.get("roc", {"fpr": [0, 1], "tpr": [0, 1]})
    ev["pr"] = _roc_pr_curves.get("pr", {"recall": [0, 1], "precision": [1, 0]})
    ev["confusion"] = _confusion_matrix
    ev["model_version"] = _model_metadata.get("model_version", "1.0-fedavg")

    return jsonify(ev)


@app.route("/api/fl/evaluation/per-class", methods=["GET"])
def fl_evaluation_per_class():
    """Per-class AUC, precision, recall, F1 breakdown."""
    per_class = _model_evaluation.get("per_class", {})
    return jsonify({"per_class": per_class, "labels": _labels})


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Visualization gallery (serve EDA figures)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/visualizations", methods=["GET"])
def get_visualizations():
    """Return list of available EDA figure URLs."""
    fig_map = {
        "disease_distribution_bar.png": {
            "title": "Disease count plot",
            "desc": "Frequency of each label across the full dataset.",
        },
        "age_vs_finding_boxplot.png": {
            "title": "Age vs. finding boxplot",
            "desc": "Spread of patient age within each diagnosis.",
        },
        "sample_xray_grid.png": {
            "title": "Sample X-ray grid",
            "desc": "Representative images per class, pre-augmentation.",
        },
        "pixel_intensity_histogram.png": {
            "title": "Pixel intensity histogram",
            "desc": "Normalized pixel value distribution after preprocessing.",
        },
        "class_imbalance_chart.png": {
            "title": "Class imbalance chart",
            "desc": "Ratio of minority to majority classes pre/post augmentation.",
        },
        "hospital_split_pie.png": {
            "title": "Hospital split pie chart",
            "desc": "Share of the dataset allocated to each virtual hospital.",
        },
    }
    available = []
    for filename, meta in fig_map.items():
        path = config.FIGURES_DIR / filename
        if path.exists():
            available.append({**meta, "url": f"/api/figures/{filename}"})
    return jsonify({"items": available})


@app.route("/api/figures/<filename>", methods=["GET"])
def serve_figure(filename):
    """Serve a saved EDA / training figure."""
    path = config.FIGURES_DIR / secure_filename(filename)
    if not path.exists():
        return jsonify({"error": "Figure not found."}), 404
    return send_file(str(path), mimetype="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max size: {config.MAX_UPLOAD_MB} MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": f"Internal server error: {e}"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
load_model_and_artifacts()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=False)
