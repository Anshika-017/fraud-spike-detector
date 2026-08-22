"""Evaluate on the held-out test set the model never saw. Reports
precision/recall/PR-AUC and picks a threshold by COST, not by whatever
gives the prettiest F1."""
import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

from data import load_data, xy, REPO_ROOT
import os
MODEL_PATH = os.path.join(REPO_ROOT, "data", "model.pkl")

# --- edit these to reflect your reasoning about real costs ---
COST_MISSED_FRAUD = 100   # avg amount lost when fraud slips through
COST_FALSE_ALARM = 5      # cost of annoying / blocking one genuine customer
# ----------------------------------------------------------------

_, test_df = load_data()
X_test, y_test = xy(test_df)

model = joblib.load(MODEL_PATH)
scores = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, scores)
pr_auc = average_precision_score(y_test, scores)

# sweep thresholds, pick the one minimizing total expected cost
n_pos = y_test.sum()
best = None
for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
    tp = r * n_pos
    fn = n_pos - tp
    fp = tp * (1 - p) / p if p > 0 else 0
    cost = fn * COST_MISSED_FRAUD + fp * COST_FALSE_ALARM
    if best is None or cost < best["cost"]:
        best = {"threshold": t, "precision": p, "recall": r, "cost": cost}

print(f"PR-AUC: {pr_auc:.4f}")
print(f"Best threshold (min cost): {best['threshold']:.4f}")
print(f"  Precision: {best['precision']:.4f}")
print(f"  Recall:    {best['recall']:.4f}")
print(f"  Est. total cost at this threshold: {best['cost']:.2f}")
