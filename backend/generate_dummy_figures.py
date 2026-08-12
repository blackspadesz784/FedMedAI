import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set directories
project_root = Path(r"c:\Users\DELL\Documents\Projects\Project-8\FedMed AI")
figures_dir = project_root / "backend" / "docs_figures"
figures_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="crest")

# Fake data
all_labels = ["No Finding", "Infiltration", "Effusion", "Atelectasis", "Pneumonia", "Cardiomegaly", "Pneumothorax", "Mass"]
counts = [1780, 980, 845, 760, 640, 512, 380, 309]
label_counts = pd.Series(counts, index=all_labels)

# 1. Disease distribution bar
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=label_counts.values, y=label_counts.index, palette="crest", ax=ax)
ax.set_title("Disease Distribution Across Dataset")
ax.set_xlabel("Number of images")
plt.tight_layout()
plt.savefig(figures_dir / "disease_distribution_bar.png", dpi=120)
plt.close()

# 2. Share of diagnoses — pie chart
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%",
       colors=sns.color_palette("crest", len(label_counts)))
ax.set_title("Share of Top Diagnoses")
plt.tight_layout()
plt.savefig(figures_dir / "disease_share_pie.png", dpi=120)
plt.close()

# 3. Age distribution
ages = np.random.normal(loc=55, scale=15, size=2000)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(ages, bins=20, kde=True, color="#2f9e68", ax=ax)
ax.set_title("Patient Age Distribution")
plt.tight_layout()
plt.savefig(figures_dir / "age_distribution.png", dpi=120)
plt.close()

# 4. Finding correlation heatmap
corr_data = np.array([
    [1.00, 0.42, 0.31, 0.18, 0.27],
    [0.42, 1.00, 0.38, 0.22, 0.15],
    [0.31, 0.38, 1.00, 0.46, 0.11],
    [0.18, 0.22, 0.46, 1.00, 0.09],
    [0.27, 0.15, 0.11, 0.09, 1.00],
])
corr_labels = ["Effusion", "Atelectasis", "Infiltration", "Pneumonia", "Cardiomegaly"]
corr_df = pd.DataFrame(corr_data, index=corr_labels, columns=corr_labels)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="crest", ax=ax)
ax.set_title("Co-occurrence Correlation Between Findings")
plt.tight_layout()
plt.savefig(figures_dir / "finding_correlation_heatmap.png", dpi=120)
plt.close()

# 5. FL progress
fl_history = {
    "round": list(range(1, 19)),
    "global_val_accuracy": [0.61,0.67,0.71,0.75,0.78,0.80,0.82,0.84,0.855,0.865,0.875,0.885,0.89,0.897,0.902,0.907,0.911,0.915],
    "global_val_loss": [0.98,0.81,0.68,0.58,0.50,0.44,0.39,0.35,0.32,0.30,0.28,0.26,0.25,0.24,0.23,0.225,0.22,0.214],
    "global_val_auc": [0.65,0.70,0.75,0.79,0.82,0.84,0.86,0.87,0.885,0.895,0.905,0.91,0.915,0.92,0.923,0.926,0.928,0.93],
}
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

print("Generated dummy figures!")
