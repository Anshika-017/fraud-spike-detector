"""Generates two plots for the README: a confusion matrix at the chosen
threshold, and the full precision-recall curve. Run this AFTER train.py."""
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

from data import load_data, xy, REPO_ROOT
MODEL_PATH = os.path.join(REPO_ROOT, "data", "model.pkl")
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# same cost values as evaluate.py — keep these in sync if you change one
COST_MISSED_FRAUD = 100
COST_FALSE_ALARM = 5

_, test_df = load_data()
X_test, y_test = xy(test_df)
model = joblib.load(MODEL_PATH)
scores = model.predict_proba(X_test)[:, 1]

# --- re-find the same cost-minimizing threshold evaluate.py picks ---
precision, recall, thresholds = precision_recall_curve(y_test, scores)
pr_auc = average_precision_score(y_test, scores)
n_pos = y_test.sum()
best = None
for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
    tp = r * n_pos
    fn = n_pos - tp
    fp = tp * (1 - p) / p if p > 0 else 0
    cost = fn * COST_MISSED_FRAUD + fp * COST_FALSE_ALARM
    if best is None or cost < best["cost"]:
        best = {"threshold": t, "cost": cost}
threshold = best["threshold"]

# --- confusion matrix at that threshold ---
y_pred = (scores >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fraud"]).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title(f"Confusion Matrix (threshold={threshold:.4f})")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
print(f"Saved plots/confusion_matrix.png")

# --- precision-recall curve ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=150)
print(f"Saved plots/pr_curve.png")
