"""Fraud classifier: cross-validated XGBoost + Random Forest ensemble."""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from data import load_data, xy, REPO_ROOT
MODEL_PATH = os.path.join(REPO_ROOT, "data", "model.pkl")

train_df, _ = load_data()
X_train, y_train = xy(train_df)

print(f"Train rows: {len(train_df)}  |  Fraud in train: {y_train.sum()} "
      f"({y_train.mean():.4%})")

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
)
xgb_params = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    xgb, xgb_params, n_iter=15, scoring="average_precision",
    cv=cv, random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
best_xgb = search.best_estimator_
print(f"Best XGBoost params: {search.best_params_}")
print(f"Best CV PR-AUC: {search.best_score_:.4f}")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, class_weight="balanced",
    random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)

model = VotingClassifier(
    estimators=[("xgb", best_xgb), ("rf", rf)], voting="soft",
)
model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
