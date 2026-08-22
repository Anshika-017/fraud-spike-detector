# Vigil

Detects individual fraudulent transactions AND flags when fraud is spiking
above the recent baseline rate — built for the Razorpay Buildathon
(AI Risk Manager track).

## What it does
1. Trains a classifier on labeled transactions (fraud vs genuine).
2. Picks a decision threshold using cost, not accuracy (a missed fraud costs
   the transaction amount; a false alarm costs one annoyed customer).
3. Buckets flagged transactions into time windows and flags a "spike" when
   the flagged rate jumps well above its own recent rolling baseline.
4. Reports precision/recall/PR-AUC on a held-out test set it never trained on.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Get the data
Download the "Credit Card Fraud Detection" dataset from Kaggle
(search: `mlg-ulb creditcardfraud`) and put `creditcard.csv` in `data/`.

Don't have it yet / just want to test the pipeline runs? Run:
```bash
python src/make_synthetic.py
```
This drops a small fake `data/creditcard.csv` with the same column shape so
every script below works immediately. Swap in the real file when you have it.

## Run it
```bash
python src/train.py      # trains model, saves to data/model.pkl
python src/evaluate.py   # precision/recall/PR-AUC on held-out test set
python src/spike_monitor.py   # rolling-window spike flags
```

## Results
- Precision: 0.5351
- Recall: 0.8133
- PR-AUC: 0.8175
- False-positive cost reasoning: missing a fraud costs the merchant the
  transaction amount; a false alarm costs one annoyed customer. Weighted
  100:5 in favor of catching fraud, so the model deliberately over-flags —
  precision is lower than recall by design, not by accident.

## Spike monitor
Test window only spans ~8 hourly buckets with no artificial fraud burst
injected, so 0 spikes flagged is the correct/honest result on this dataset,
not a failure of the detector.

## What broke
Started with a plain logistic regression baseline to get one clean
end-to-end pipeline working first — train, evaluate, save — before
touching anything complex. Once that ran correctly, upgraded to a
cross-validated XGBoost + Random Forest ensemble with engineered features
(log-transformed amount, hour-of-day) and cost-based threshold selection
instead of accuracy. That two-stage approach (dumb-and-working before
smart-and-optimized) is what let me catch a rolling-window spike-detection
bug against a known-working baseline instead of debugging two new things
at once.

Separately: xgboost's 48MB install kept timing out on a slow connection —
three failed attempts (ReadTimeoutError). Fixed with
`--timeout 300 --no-cache-dir`. Also lost a full training run when my
machine restarted mid-hyperparameter-search; no code issue, just reran it.
