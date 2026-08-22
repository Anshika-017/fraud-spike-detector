"""Load and split the transaction data. Time-ordered split, not random —
fraud detection is a forecasting problem: you only ever have the past to
predict the future, so the test set is the LAST slice in time, never a
random shuffle."""
import os
import numpy as np
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "LogAmount", "Hour"]
LABEL = "Class"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO_ROOT, "data", "creditcard.csv")


def _engineer(df):
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] % 86400) // 3600
    return df


def load_data(path=DEFAULT_PATH, test_frac=0.2):
    df = pd.read_csv(path).sort_values("Time").reset_index(drop=True)
    df = _engineer(df)
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def xy(df):
    return df[FEATURES], df[LABEL]
