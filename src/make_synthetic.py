"""Creates a tiny fake creditcard.csv so the pipeline runs before you have
the real Kaggle dataset. Same columns: Time, V1..V28, Amount, Class."""
import numpy as np
import pandas as pd

n_rows = 20000
fraud_rate = 0.0017
rng = np.random.default_rng(42)

n = n_rows
n_fraud = max(1, int(n * fraud_rate))
y = np.zeros(n, dtype=int)
y[:n_fraud] = 1
rng.shuffle(y)

data = {"Time": np.sort(rng.integers(0, 172800, n))}
for i in range(1, 29):
    shift = rng.normal(0, 0.6) if i % 3 == 0 else 0  # fraud rows drift on some features
    data[f"V{i}"] = rng.normal(0, 1, n) + y * shift
data["Amount"] = np.round(np.abs(rng.normal(80, 120, n)) + y * rng.normal(50, 30, n), 2)
data["Class"] = y

df = pd.DataFrame(data)
import os
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "creditcard.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with {n} rows, {n_fraud} fraud ({fraud_rate:.3%})")
