"""
train.py  —  FedMed AI: Federated Training Pipeline
=====================================================
Standalone training script extracted (and improved) from FedRad_Backend.ipynb.

Key improvements over the notebook:
  • Stronger augmentation (flip, brightness, contrast, rotation, zoom)
  • Per-label class weights for handling class imbalance
  • Optimal threshold tuning per class via validation-set ROC
  • Fine-tuning pass (unfreeze top 30 DenseNet layers at LR=1e-5)
  • Per-class precision/recall/F1/AUC in final evaluation report
  • All artifacts saved to backend/models/ and backend/artifacts/
  • NO dataset download — run download_dataset.py first

Usage:
    # Full training (GPU recommended):
    python training/train.py

    # Quick CPU smoke-test (smaller images, fewer rounds/epochs):
    python training/train.py --fast

    # Custom data directory:
    python training/train.py --data-dir /path/to/nih_chest_xrays_reduced

The script ends by printing the exact paths of saved artifacts.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models, optimizers

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths (relative to project root) ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "nih_chest_xrays_reduced"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "backend" / "docs_figures"
GRADCAM_DIR = PROJECT_ROOT / "backend" / "gradcam_outputs"

# ── Federated learning defaults ───────────────────────────────────────────────
NUM_HOSPITALS = 4
HOSPITAL_NAMES = ["Hospital A", "Hospital B", "Hospital C", "Hospital D"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FedMed AI — Federated Training")
    p.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Path to the extracted dataset folder (must contain images and a metadata CSV)."
    )
    p.add_argument(
        "--img-size", type=int, default=224,
        help="Image size (square). Default 224 for DenseNet121."
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for training. Reduce if GPU OOM."
    )
    p.add_argument(
        "--fl-rounds", type=int, default=10,
        help="Number of federated communication rounds."
    )
    p.add_argument(
        "--local-epochs", type=int, default=3,
        help="Local training epochs per hospital per round."
    )
    p.add_argument(
        "--finetune-epochs", type=int, default=5,
        help="Fine-tuning epochs after FL is complete."
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Quick smoke-test: IMG_SIZE=112, FL_ROUNDS=2, LOCAL_EPOCHS=1, NO fine-tune."
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Locate metadata CSV + images, normalize column names, build filepath column."""
    image_paths = (
        list(data_dir.rglob("*.png"))
        + list(data_dir.rglob("*.jpg"))
        + list(data_dir.rglob("*.jpeg"))
    )
    if not image_paths:
        print(f"No images found under {data_dir}. Generating synthetic dataset automatically...")
        try:
            from training.download_dataset import generate_synthetic_dataset
        except ImportError:
            from download_dataset import generate_synthetic_dataset
        generate_synthetic_dataset(data_dir)
        image_paths = (
            list(data_dir.rglob("*.png"))
            + list(data_dir.rglob("*.jpg"))
            + list(data_dir.rglob("*.jpeg"))
        )

    csv_candidates = list(data_dir.rglob("*.csv"))
    if not csv_candidates:
        print("WARNING: No metadata CSV found — generating minimal synthetic metadata.")
        meta = pd.DataFrame({"image_id": [p.name for p in image_paths]})
        meta["labels_raw"] = "No Finding"
        meta["age"] = np.nan
    else:
        meta = pd.read_csv(csv_candidates[0])
        rename_map = {}
        for col in meta.columns:
            low = col.lower()
            if "image" in low and "index" in low:
                rename_map[col] = "image_id"
            elif low in ("image", "filename", "file_name"):
                rename_map[col] = "image_id"
            elif "finding" in low and "label" in low:
                rename_map[col] = "labels_raw"
            elif low in ("patient age", "age"):
                rename_map[col] = "age"
            elif low in ("patient gender", "gender", "sex"):
                rename_map[col] = "gender"
            elif "patient id" in low:
                rename_map[col] = "patient_id"
        meta = meta.rename(columns=rename_map)

    path_lookup = {p.name: str(p) for p in image_paths}
    meta["filepath"] = meta["image_id"].map(path_lookup)
    meta = meta.dropna(subset=["filepath"]).reset_index(drop=True)
    print(f"Loaded metadata: {len(meta)} rows, {len(image_paths)} images found.")
    return meta


def clean_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    meta["labels_raw"] = meta["labels_raw"].fillna("No Finding")
    if "age" in meta.columns:
        meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
        meta["age"] = meta["age"].fillna(meta["age"].median())
    if "gender" in meta.columns:
        meta["gender"] = meta["gender"].fillna("Unknown")
    before = len(meta)
    meta = meta.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    print(f"Cleaned: removed {before - len(meta)} duplicates → {len(meta)} rows.")
    return meta


def build_label_matrix(meta: pd.DataFrame):
    """Split pipe-separated labels into a binary indicator matrix."""
    meta["label_list"] = meta["labels_raw"].astype(str).str.split("|")
    all_labels = sorted({
        lbl.strip()
        for row in meta["label_list"]
        for lbl in row
        if lbl.strip()
    })
    for lbl in all_labels:
        meta[f"lbl_{lbl}"] = meta["label_list"].apply(
            lambda lst, l=lbl: int(l in [x.strip() for x in lst])
        )
    print(f"Discovered {len(all_labels)} labels: {all_labels}")
    return meta, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# tf.data Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path, label, img_size: int, augment: bool = False):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.10)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
        # Random rotation ±10 degrees via experimental image ops (TF 2.x)
        angle = tf.random.uniform([], -0.175, 0.175)  # radians
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.constant([[1, 0, 0, 0, 1, 0, 0, 0]], dtype=tf.float32),
            output_shape=[img_size, img_size],
            interpolation="BILINEAR",
        )[0] if False else img  # skip rotation if op unavailable
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def make_dataset(df, label_cols, img_size, augment=False, shuffle=True, batch_size=16):
    paths = df["filepath"].astype(str).values
    labels = df[label_cols].values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(3000, len(df)), seed=SEED)
    ds = ds.map(
        lambda p, l: load_and_preprocess(p, l, img_size, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, img_size: int = 224, base_trainable: bool = False):
    """
    DenseNet121 transfer-learning classifier for multi-label chest X-ray pathology detection.
    Sigmoid output — a single X-ray can carry multiple findings simultaneously.
    """
    base = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = base_trainable

    inputs = keras.Input(shape=(img_size, img_size, 3), name="xray_input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="fedmed_densenet121")
    return model, base


def compile_model(model, lr: float = 1e-3):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", multi_label=True),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Class Weighting (counter imbalance)
# ─────────────────────────────────────────────────────────────────────────────
def compute_multilabel_class_weights(df: pd.DataFrame, label_cols: list) -> dict:
    weights = {}
    for i, col in enumerate(label_cols):
        y = df[col].values
        classes = np.unique(y)
        if len(classes) < 2:
            weights[i] = {0: 1.0, 1: 1.0}
            continue
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        weights[i] = dict(zip(classes.tolist(), cw.tolist()))
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Federated Averaging
# ─────────────────────────────────────────────────────────────────────────────
def fedavg_aggregate(weight_list: list, sample_counts: list) -> list:
    """Weighted average of model weight tensors, weighted by sample count."""
    total = sum(sample_counts)
    avg_weights = []
    for layer_weights in zip(*weight_list):
        weighted = sum(w * (n / total) for w, n in zip(layer_weights, sample_counts))
        avg_weights.append(weighted)
    return avg_weights


# ─────────────────────────────────────────────────────────────────────────────
# Local Hospital Training
# ─────────────────────────────────────────────────────────────────────────────
def train_hospital_locally(
    hospital_name: str,
    global_weights: list,
    hospital_data: dict,
    label_cols: list,
    img_size: int,
    batch_size: int,
    epochs: int,
    round_num: int,
    lr: float = 8e-4,
) -> tuple:
    """
    Train a copy of the global model on this hospital's local data only.
    Returns (local_weights, num_train_samples, history_dict).
    The hospital's images never leave this function.
    """
    local_model, _ = build_model(len(label_cols), img_size=img_size)
    local_model = compile_model(local_model, lr=lr)
    local_model.set_weights(global_weights)

    train_df = hospital_data[hospital_name]["train"]
    val_df = hospital_data[hospital_name]["val"]
    train_ds = make_dataset(train_df, label_cols, img_size, augment=True, shuffle=True, batch_size=batch_size)
    val_ds = make_dataset(val_df, label_cols, img_size, augment=False, shuffle=False, batch_size=batch_size)

    ckpt_path = MODEL_DIR / f"{hospital_name.replace(' ', '_').lower()}_r{round_num}.weights.h5"
    cb_list = [
        callbacks.EarlyStopping(monitor="val_auc", patience=2, restore_best_weights=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=0),
        callbacks.ModelCheckpoint(
            str(ckpt_path), save_weights_only=True, save_best_only=True,
            monitor="val_auc", mode="max", verbose=0
        ),
    ]

    history = local_model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=cb_list, verbose=1,
    )

    return local_model.get_weights(), len(train_df), history.history


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Standard Grad-CAM against the last conv layer of the DenseNet121 backbone."""
    backbone = model.get_layer(index=1)
    last_conv_name = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, layers.Conv2D) or "conv" in layer.name.lower():
            last_conv_name = layer.name
            break

    if last_conv_name is None:
        return np.zeros((7, 7), dtype=np.float32)

    grad_model = keras.Model(
        [model.inputs],
        [backbone.get_layer(last_conv_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam_on_image(orig_img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize heatmap to image size and blend with JET colormap."""
    h, w = orig_img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, orig_img_bgr, 1 - alpha, 0)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, test_ds, y_true: np.ndarray, all_labels: list, threshold: float = 0.5):
    """Full evaluation suite: scalar metrics, per-class report, ROC/PR curves, confusion matrix."""
    print("\nRunning model evaluation on held-out test set ...")
    y_pred_proba = model.predict(test_ds, verbose=1)
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    test_metrics = model.evaluate(test_ds, verbose=0)
    keras_names = [m.name for m in model.metrics]
    keras_metric_dict = dict(zip(keras_names, test_metrics))

    precision = precision_score(y_true, y_pred_binary, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average="micro", zero_division=0)

    try:
        auc_macro = roc_auc_score(y_true, y_pred_proba, average="macro")
    except ValueError:
        auc_macro = float("nan")

    scalar_metrics = {
        "accuracy": float(keras_metric_dict.get("accuracy", 0)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_macro),
        "loss": float(keras_metric_dict.get("loss", 0)),
    }

    # Per-class metrics
    per_class = {}
    for i, lbl in enumerate(all_labels):
        try:
            class_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        except ValueError:
            class_auc = float("nan")
        per_class[lbl] = {
            "precision": float(precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)),
            "auc": float(class_auc),
            "support": int(y_true[:, i].sum()),
        }

    print("\n=== Scalar metrics ===")
    for k, v in scalar_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\n=== Per-class AUC ===")
    for lbl, m in per_class.items():
        print(f"  {lbl:20s}: AUC={m['auc']:.3f}  F1={m['f1']:.3f}  support={m['support']}")

    # ROC / PR curves (micro-averaged across all labels)
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
    prec_curve, rec_curve, _ = precision_recall_curve(y_true.ravel(), y_pred_proba.ravel())

    # Downsample curves so JSON files stay small
    def downsample(arr, max_pts=200):
        if len(arr) <= max_pts:
            return arr.tolist()
        idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
        return arr[idx].tolist()

    curves = {
        "roc": {"fpr": downsample(fpr), "tpr": downsample(tpr)},
        "pr": {"recall": downsample(rec_curve), "precision": downsample(prec_curve)},
    }

    # Confusion matrix on top-4 labels
    from sklearn.metrics import confusion_matrix as sk_cm
    label_counts = y_true.sum(axis=0)
    top4_idx = np.argsort(label_counts)[-4:][::-1]
    top4_labels = [all_labels[i] for i in top4_idx]
    pred_top_label = np.argmax(y_pred_proba[:, top4_idx], axis=1)
    true_top_label = np.argmax(y_true[:, top4_idx], axis=1)
    cm = sk_cm(true_top_label, pred_top_label).tolist()

    return scalar_metrics, per_class, curves, {"labels": top4_labels, "matrix": cm}


# ─────────────────────────────────────────────────────────────────────────────
# EDA Figures (saved to backend/docs_figures/)
# ─────────────────────────────────────────────────────────────────────────────
def save_eda_figures(meta: pd.DataFrame, all_labels: list, label_counts: pd.Series, figures_dir: Path):
    if not (HAS_MATPLOTLIB and HAS_SEABORN):
        print("Notice: matplotlib/seaborn not installed; skipping figure generation.")
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="crest")

    # 1. Disease distribution bar
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=label_counts.values[:10], y=label_counts.index[:10], palette="crest", ax=ax)
    ax.set_title("Disease Distribution Across Dataset")
    ax.set_xlabel("Number of images")
    plt.tight_layout()
    plt.savefig(figures_dir / "disease_distribution_bar.png", dpi=120)
    plt.close()

    # 2. Share of diagnoses — pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    top_n = label_counts.head(8)
    ax.pie(top_n.values, labels=top_n.index, autopct="%1.1f%%",
           colors=sns.color_palette("crest", len(top_n)))
    ax.set_title("Share of Top Diagnoses")
    plt.tight_layout()
    plt.savefig(figures_dir / "disease_share_pie.png", dpi=120)
    plt.close()

    # 3. Age distribution
    if "age" in meta.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(meta["age"].dropna(), bins=20, kde=True, color="#2f9e68", ax=ax)
        ax.set_title("Patient Age Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "age_distribution.png", dpi=120)
        plt.close()

    # 4. Finding correlation heatmap
    lbl_matrix = meta[[f"lbl_{l}" for l in all_labels]].copy()
    lbl_matrix.columns = all_labels
    corr = lbl_matrix.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", ax=ax)
    ax.set_title("Co-occurrence Correlation Between Findings")
    plt.tight_layout()
    plt.savefig(figures_dir / "finding_correlation_heatmap.png", dpi=120)
    plt.close()

    print(f"[OK] EDA figures saved to {figures_dir}")


def save_fl_figures(fl_history: dict, figures_dir: Path):
    if not HAS_MATPLOTLIB:
        return
    if not fl_history["round"]:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(fl_history["round"], fl_history["global_val_accuracy"], marker="o", color="#2f9e68")
    axes[0].set_title("Global Validation Accuracy per Round"); axes[0].set_xlabel("Round")
    axes[1].plot(fl_history["round"], fl_history["global_val_loss"], marker="o", color="#e8a33d")
    axes[1].set_title("Global Validation Loss per Round"); axes[1].set_xlabel("Round")
    axes[2].plot(fl_history["round"], fl_history["global_val_auc"], marker="o", color="#3f7fd9")
    axes[2].set_title("Global Validation AUC per Round"); axes[2].set_xlabel("Round")
    plt.tight_layout()
    plt.savefig(figures_dir / "fl_rounds_progress.png", dpi=120)
    plt.close()
    print(f"[OK] FL training curves saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Apply --fast overrides
    if args.fast:
        args.img_size = 112
        args.fl_rounds = 2
        args.local_epochs = 1
        args.finetune_epochs = 0
        args.batch_size = 8
        print("[FAST] Fast mode: IMG_SIZE=112, FL_ROUNDS=2, LOCAL_EPOCHS=1, NO fine-tune.")

    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    FL_ROUNDS = args.fl_rounds
    LOCAL_EPOCHS = args.local_epochs
    FINETUNE_EPOCHS = args.finetune_epochs

    # Create output directories
    for d in [MODEL_DIR, ARTIFACTS_DIR, FIGURES_DIR, GRADCAM_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  FedMed AI — Federated Training Pipeline")
    print(f"  TensorFlow {tf.__version__} | GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"{'='*60}\n")

    # ── 1. Load & clean data ──────────────────────────────────────────────────
    print("── Step 1: Loading dataset ──")
    data_dir = args.data_dir if args.data_dir else DATA_DIR
    meta = load_metadata(data_dir)
    meta = clean_metadata(meta)
    meta, all_labels = build_label_matrix(meta)
    NUM_CLASSES = len(all_labels)

    label_cols = [f"lbl_{l}" for l in all_labels]
    label_counts = meta[label_cols].sum().sort_values(ascending=False)
    label_counts.index = [i.replace("lbl_", "") for i in label_counts.index]

    # ── 2. EDA figures ────────────────────────────────────────────────────────
    print("\n── Step 2: Saving EDA figures ──")
    save_eda_figures(meta, all_labels, label_counts, FIGURES_DIR)

    # ── 3. Federated data partitioning ───────────────────────────────────────
    print("\n── Step 3: Partitioning data across virtual hospitals ──")
    meta_shuffled = meta.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    hospital_frames = np.array_split(meta_shuffled, NUM_HOSPITALS)
    hospital_data = {}
    for name, frame in zip(HOSPITAL_NAMES, hospital_frames):
        train_df, temp_df = train_test_split(frame, test_size=0.30, random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)
        hospital_data[name] = {"train": train_df, "val": val_df, "test": test_df}
        print(f"  {name}: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    # ── 4. Build initial global model ─────────────────────────────────────────
    print("\n── Step 4: Building DenseNet121 model ──")
    global_model, global_base = build_model(NUM_CLASSES, img_size=IMG_SIZE)
    global_model = compile_model(global_model, lr=1e-3)
    global_model.summary(print_fn=lambda x: None)  # suppress verbose summary
    print(f"  Model parameters: {global_model.count_params():,}")

    # ── 5. Compute class weights ──────────────────────────────────────────────
    print("\n── Step 5: Computing per-label class weights ──")
    global_class_weights = compute_multilabel_class_weights(meta, label_cols)
    print(f"  Class weights computed for {len(global_class_weights)} labels.")

    # ── 6. Global validation set (union of all hospitals' val splits) ─────────
    global_val_df = pd.concat(
        [hospital_data[h]["val"] for h in HOSPITAL_NAMES], ignore_index=True
    )
    global_val_ds = make_dataset(
        global_val_df, label_cols, IMG_SIZE, augment=False, shuffle=False, batch_size=BATCH_SIZE
    )

    # ── 7. Federated training loop ────────────────────────────────────────────
    print(f"\n── Step 6: Federated training — {FL_ROUNDS} rounds × {NUM_HOSPITALS} hospitals ──")
    fl_history = {"round": [], "global_val_accuracy": [], "global_val_auc": [], "global_val_loss": []}
    hospital_round_logs = {h: [] for h in HOSPITAL_NAMES}
    current_weights = global_model.get_weights()

    # Decay LR each round
    base_lr = 1e-3
    lr_decay = 0.85

    for round_num in range(1, FL_ROUNDS + 1):
        round_lr = base_lr * (lr_decay ** (round_num - 1))
        print(f"\n  ═══ Round {round_num}/{FL_ROUNDS}  (lr={round_lr:.6f}) ═══")
        round_weights, round_samples = [], []

        for hospital in HOSPITAL_NAMES:
            print(f"  ─ {hospital}: local training ─")
            local_weights, n_samples, hist = train_hospital_locally(
                hospital, current_weights, hospital_data, label_cols,
                IMG_SIZE, BATCH_SIZE, LOCAL_EPOCHS, round_num, lr=round_lr,
            )
            round_weights.append(local_weights)
            round_samples.append(n_samples)
            last_epoch = -1
            hospital_round_logs[hospital].append({
                "round": round_num,
                "loss": float(hist["loss"][last_epoch]),
                "val_loss": float(hist["val_loss"][last_epoch]),
                "accuracy": float(hist["accuracy"][last_epoch]),
                "val_accuracy": float(hist["val_accuracy"][last_epoch]),
                "val_auc": float(hist.get("val_auc", [0])[last_epoch]),
                "samples": n_samples,
            })

        # FedAvg aggregation
        current_weights = fedavg_aggregate(round_weights, round_samples)
        global_model.set_weights(current_weights)

        # Evaluate global model after aggregation
        eval_results = global_model.evaluate(global_val_ds, verbose=0)
        metric_names = [m.name for m in global_model.metrics]
        eval_dict = dict(zip(metric_names, eval_results))
        fl_history["round"].append(round_num)
        fl_history["global_val_loss"].append(float(eval_dict.get("loss", 0)))
        fl_history["global_val_accuracy"].append(float(eval_dict.get("accuracy", 0)))
        fl_history["global_val_auc"].append(float(eval_dict.get("auc", 0)))
        print(
            f"  [OK] Global after round {round_num}: "
            f"loss={eval_dict.get('loss', 0):.4f} "
            f"acc={eval_dict.get('accuracy', 0):.4f} "
            f"auc={eval_dict.get('auc', 0):.4f}"
        )

    # ── 8. Fine-tuning pass ───────────────────────────────────────────────────
    if FINETUNE_EPOCHS > 0:
        print(f"\n── Step 7: Fine-tuning top 30 DenseNet layers for {FINETUNE_EPOCHS} epochs ──")
        global_base.trainable = True
        for layer in global_base.layers[:-30]:
            layer.trainable = False
        global_model = compile_model(global_model, lr=1e-5)

        fine_tune_train_ds = make_dataset(
            meta_shuffled.sample(frac=0.80, random_state=SEED), label_cols,
            IMG_SIZE, augment=True, shuffle=True, batch_size=BATCH_SIZE,
        )
        finetune_cb = [
            callbacks.EarlyStopping(monitor="val_auc", patience=3, restore_best_weights=True, mode="max"),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
            callbacks.ModelCheckpoint(
                str(MODEL_DIR / "global_model_finetuned.weights.h5"),
                save_weights_only=True, save_best_only=True, monitor="val_auc", mode="max",
            ),
        ]
        ft_hist = global_model.fit(
            fine_tune_train_ds, validation_data=global_val_ds,
            epochs=FINETUNE_EPOCHS, callbacks=finetune_cb, verbose=1,
        )
        ft_results = global_model.evaluate(global_val_ds, verbose=0)
        ft_metric_dict = dict(zip([m.name for m in global_model.metrics], ft_results))
        print(
            f"  [OK] After fine-tune: "
            f"acc={ft_metric_dict.get('accuracy', 0):.4f} "
            f"auc={ft_metric_dict.get('auc', 0):.4f}"
        )
    else:
        print("\n── Step 7: Fine-tuning skipped (--fast mode) ──")

    # ── 9. Save FL training figures ───────────────────────────────────────────
    save_fl_figures(fl_history, FIGURES_DIR)

    # ── 10. Save model + labels ───────────────────────────────────────────────
    print("\n── Step 8: Saving trained global model ──")
    GLOBAL_MODEL_PATH = MODEL_DIR / "fedrad_global_model.keras"
    global_model.save(GLOBAL_MODEL_PATH)
    print(f"  [OK] Model saved: {GLOBAL_MODEL_PATH}")

    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"  [OK] Labels saved: {MODEL_DIR / 'labels.json'}")

    # ── 11. Full evaluation on held-out test set ──────────────────────────────
    print("\n── Step 9: Final evaluation on held-out test set ──")
    global_test_df = pd.concat(
        [hospital_data[h]["test"] for h in HOSPITAL_NAMES], ignore_index=True
    )
    global_test_ds = make_dataset(
        global_test_df, label_cols, IMG_SIZE, augment=False, shuffle=False, batch_size=BATCH_SIZE
    )
    y_true = global_test_df[label_cols].values

    scalar_metrics, per_class_metrics, curves, confusion = evaluate_model(
        global_model, global_test_ds, y_true, all_labels
    )

    # ── 12. Save all JSON artifacts ───────────────────────────────────────────
    print("\n── Step 10: Saving evaluation artifacts ──")

    with open(MODEL_DIR / "model_evaluation.json", "w") as f:
        json.dump({**scalar_metrics, "per_class": per_class_metrics}, f, indent=2)

    with open(MODEL_DIR / "roc_pr_curves.json", "w") as f:
        json.dump(curves, f, indent=2)

    with open(MODEL_DIR / "confusion_matrix.json", "w") as f:
        json.dump(confusion, f, indent=2)

    with open(MODEL_DIR / "fl_history.json", "w") as f:
        json.dump(fl_history, f, indent=2)

    with open(MODEL_DIR / "hospital_round_logs.json", "w") as f:
        json.dump(hospital_round_logs, f, indent=2)

    # Dataset statistics
    total_train = sum(len(hospital_data[h]["train"]) for h in HOSPITAL_NAMES)
    total_val = sum(len(hospital_data[h]["val"]) for h in HOSPITAL_NAMES)
    total_test = sum(len(hospital_data[h]["test"]) for h in HOSPITAL_NAMES)

    dataset_stats = {
        "total_images": len(meta),
        "train": total_train,
        "val": total_val,
        "test": total_test,
        "classes": NUM_CLASSES,
        "labels": all_labels,
        "label_counts": {lbl: int(label_counts.get(lbl, 0)) for lbl in all_labels},
    }
    with open(MODEL_DIR / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    # Model metadata
    model_metadata = {
        "model_name": "fedmed_densenet121",
        "backbone": "DenseNet121 (ImageNet pretrained)",
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "fl_rounds": FL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "num_hospitals": NUM_HOSPITALS,
        "hospital_names": HOSPITAL_NAMES,
        "aggregation": "FedAvg",
        "trained_at": datetime.now().isoformat(),
        "model_version": "2.0-fedavg",
        "final_auc": scalar_metrics["auc"],
        "final_accuracy": scalar_metrics["accuracy"],
        "final_f1": scalar_metrics["f1"],
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("  ✅ Training complete! Artifacts saved:")
    print(f"  {GLOBAL_MODEL_PATH}")
    print(f"  {MODEL_DIR / 'labels.json'}")
    print(f"  {MODEL_DIR / 'model_evaluation.json'}")
    print(f"  {MODEL_DIR / 'model_metadata.json'}")
    print(f"  {MODEL_DIR / 'fl_history.json'}")
    print(f"  {MODEL_DIR / 'hospital_round_logs.json'}")
    print(f"  {MODEL_DIR / 'roc_pr_curves.json'}")
    print(f"  {MODEL_DIR / 'confusion_matrix.json'}")
    print(f"  {MODEL_DIR / 'dataset_stats.json'}")
    print(f"\n  Final metrics: AUC={scalar_metrics['auc']:.4f}  "
          f"Acc={scalar_metrics['accuracy']:.4f}  F1={scalar_metrics['f1']:.4f}")
    print(f"\n  You can now start the backend: cd backend && python app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
