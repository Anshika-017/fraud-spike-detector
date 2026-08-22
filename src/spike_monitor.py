"""The 'spike' part: bucket flagged transactions into time windows and
alert when the flagged rate jumps well above its own recent rolling
baseline. This is what makes it a monitor, not just a per-row classifier."""
import joblib
import pandas as pd

from data import load_data, xy, REPO_ROOT
import os
MODEL_PATH = os.path.join(REPO_ROOT, "data", "model.pkl")

WINDOW_SECONDS = 3600     # bucket size (1 hour, since Time is in seconds)
ROLLING_WINDOWS = 6       # how many past buckets count as "recent baseline"
Z_THRESHOLD = 3.0         # flag a bucket if it's this many std-devs above baseline

_, test_df = load_data()
X_test, _ = xy(test_df)
model = joblib.load(MODEL_PATH)

df = test_df.copy()
df["risk_score"] = model.predict_proba(X_test)[:, 1]
df["bucket"] = (df["Time"] // WINDOW_SECONDS).astype(int)

# a "flagged" transaction uses the threshold you picked in evaluate.py —
# hardcoded here for now, wire it up to that output once you have it
FLAG_THRESHOLD = 0.5
df["flagged"] = df["risk_score"] >= FLAG_THRESHOLD

per_bucket = df.groupby("bucket")["flagged"].agg(["sum", "count"])
per_bucket["rate"] = per_bucket["sum"] / per_bucket["count"]

rolling_mean = per_bucket["rate"].rolling(ROLLING_WINDOWS, min_periods=2).mean()
rolling_std = per_bucket["rate"].rolling(ROLLING_WINDOWS, min_periods=2).std()
per_bucket["z"] = (per_bucket["rate"] - rolling_mean) / rolling_std.replace(0, 1e-9)
per_bucket["spike"] = per_bucket["z"] >= Z_THRESHOLD

spikes = per_bucket[per_bucket["spike"]]
print(f"Buckets analyzed: {len(per_bucket)}")
print(f"Spikes flagged: {len(spikes)}")
if len(spikes):
    print(spikes[["sum", "count", "rate", "z"]])
