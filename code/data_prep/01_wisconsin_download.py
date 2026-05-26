"""Download and preprocess the Wisconsin Diagnostic Breast Cancer dataset.

Splits:
    Train (60 %)  -> 341 samples  -- model fitting
    Val   (20 %)  ->  114 samples  -- calibration fitting + conformal calibration
    Test  (20 %)  ->  114 samples  -- held-out evaluation

Outputs (all under second_paper/data/wisconsin/):
    X_train.csv, X_val.csv, X_test.csv
    y_train.csv, y_val.csv, y_test.csv
    scaler.pkl              -- StandardScaler fitted on X_train only
    feature_names.txt       -- ordered list of the 30 feature names
    train_stats.csv         -- per-feature median and std from X_train (raw, pre-scale)
                              used by 02_mimic_extract.py for MIMIC harmonisation
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]          # second_paper/
OUT  = ROOT / "data" / "wisconsin"
OUT.mkdir(parents=True, exist_ok=True)

# -- 1. Download ---------------------------------------------------------------
print("Fetching Wisconsin Diagnostic Breast Cancer (UCI id=17) …")
bc   = fetch_ucirepo(id=17)
X    = bc.data.features.copy()
y_raw = bc.data.targets.squeeze()

# Encode: Malignant -> 1, Benign -> 0
label_map = {"M": 1, "B": 0}
y = y_raw.map(label_map)
if y.isna().any():
    raise ValueError(f"Unexpected label values: {y_raw.unique()}")

print(f"Dataset shape: {X.shape}  |  Malignant: {y.sum()}  Benign: {(y==0).sum()}")

feature_names = list(X.columns)
(OUT / "feature_names.txt").write_text("\n".join(feature_names))

# -- 2. Train / Val / Test split -----------------------------------------------
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv   # 0.25 × 0.80 = 0.20
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# -- 3. Save raw training statistics (for MIMIC feature imputation) ------------
train_stats = pd.DataFrame({
    "feature":  feature_names,
    "median":   X_train.median().values,
    "std":      X_train.std().values,
    "mean":     X_train.mean().values,
    "min":      X_train.min().values,
    "max":      X_train.max().values,
})
train_stats.to_csv(OUT / "train_stats.csv", index=False)
print(f"train_stats.csv saved  ({len(train_stats)} features)")

# -- 4. Scale (fit on train only) ----------------------------------------------
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                          columns=feature_names, index=X_train.index)
X_val_s   = pd.DataFrame(scaler.transform(X_val),
                          columns=feature_names, index=X_val.index)
X_test_s  = pd.DataFrame(scaler.transform(X_test),
                          columns=feature_names, index=X_test.index)

# -- 5. Save -------------------------------------------------------------------
X_train_s.to_csv(OUT / "X_train.csv", index=False)
X_val_s.to_csv(  OUT / "X_val.csv",   index=False)
X_test_s.to_csv( OUT / "X_test.csv",  index=False)

y_train.reset_index(drop=True).to_csv(OUT / "y_train.csv", index=False)
y_val.reset_index(drop=True).to_csv(  OUT / "y_val.csv",   index=False)
y_test.reset_index(drop=True).to_csv( OUT / "y_test.csv",  index=False)

joblib.dump(scaler, OUT / "scaler.pkl")

print("All Wisconsin files saved to", OUT)
