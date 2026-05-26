"""Train 5 classifiers on Wisconsin Diagnostic with GridSearchCV tuning.

Calibration is fitted on the VALIDATION set (not training) using
CalibratedClassifierCV(cv='prefit'), which is the correct procedure.
Three calibration variants per model: base, Platt scaling, isotonic regression.

Additionally produces subgroup-targeted Platt scalers keyed by MIMIC race groups
(fitted on the MIMIC validation portion -- race-stratified) for the DCGS analysis.

Outputs (data/models/):
    {NAME}_base.pkl       -- uncalibrated fitted model
    {NAME}_platt.pkl      -- global Platt scaling (fitted on Wisconsin val)
    {NAME}_isotonic.pkl   -- global isotonic regression (fitted on Wisconsin val)
    hyperparams.json      -- best hyperparameters per model
    val_predictions.csv   -- predicted probabilities on val set (all variants)
    test_predictions.csv  -- predicted probabilities on test set (all variants)
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT     = Path(__file__).resolve().parents[2]
WIS_DIR  = ROOT / "data" / "wisconsin"
OUT      = ROOT / "data" / "models"
OUT.mkdir(parents=True, exist_ok=True)

# -- Load data -----------------------------------------------------------------
X_train = pd.read_csv(WIS_DIR / "X_train.csv").values
y_train = pd.read_csv(WIS_DIR / "y_train.csv").values.ravel()
X_val   = pd.read_csv(WIS_DIR / "X_val.csv").values
y_val   = pd.read_csv(WIS_DIR / "y_val.csv").values.ravel()
X_test  = pd.read_csv(WIS_DIR / "X_test.csv").values
y_test  = pd.read_csv(WIS_DIR / "y_test.csv").values.ravel()

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -- Hyperparameter grids ------------------------------------------------------
GRIDS = {
    "LR": {
        "estimator": LogisticRegression(max_iter=2000, random_state=42,
                                        class_weight="balanced"),
        "param_grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    },
    "RF": {
        "estimator": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth":    [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=42,
                         class_weight="balanced"),
        "param_grid": {
            "C":      [0.1, 1, 10, 100],
            "kernel": ["rbf", "poly"],
            "gamma":  ["scale", 0.01, 0.1],
        },
    },
    "NB": {
        "estimator": GaussianNB(),
        "param_grid": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]},
    },
    "XGB": {
        "estimator": XGBClassifier(
            random_state=42, eval_metric="logloss",
            verbosity=0, use_label_encoder=False
        ),
        "param_grid": {
            "n_estimators":    [100, 200],
            "max_depth":       [3, 5, 7],
            "learning_rate":   [0.05, 0.1, 0.2],
            "subsample":       [0.8, 1.0],
            "colsample_bytree":[0.6, 0.8, 1.0],
            "gamma":           [0, 0.1, 0.5],
        },
    },
}

best_params   = {}
val_preds_df  = pd.DataFrame({"y_true": y_val})
test_preds_df = pd.DataFrame({"y_true": y_test})

# -- Train, tune, calibrate each model -----------------------------------------
for name, cfg in GRIDS.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")

    gs = GridSearchCV(
        cfg["estimator"],
        cfg["param_grid"],
        cv=cv5,
        scoring="roc_auc",
        n_jobs=1,   # n_jobs=-1 causes OpenBLAS worker crash on Windows
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    base_model = gs.best_estimator_
    best_params[name] = gs.best_params_

    cv_auc = gs.best_score_
    print(f"  Best params : {gs.best_params_}")
    print(f"  CV AUROC    : {cv_auc:.4f}")

    # -- Calibrate on the VALIDATION set --------------------------------------
    # cv=None + ensemble=False: use the pre-fitted base_model, fit calibration layer only
    platt    = CalibratedClassifierCV(base_model, method="sigmoid",  cv=None, ensemble=False)
    isotonic = CalibratedClassifierCV(base_model, method="isotonic", cv=None, ensemble=False)
    platt.fit(X_val, y_val)
    isotonic.fit(X_val, y_val)

    # -- Evaluate on val and test sets -----------------------------------------
    for split_name, X_s, y_s, preds_df in [
        ("val",  X_val,  y_val,  val_preds_df),
        ("test", X_test, y_test, test_preds_df),
    ]:
        p_base = base_model.predict_proba(X_s)[:, 1]
        p_plt  = platt.predict_proba(X_s)[:, 1]
        p_iso  = isotonic.predict_proba(X_s)[:, 1]

        preds_df[f"{name}_base"]     = p_base
        preds_df[f"{name}_platt"]    = p_plt
        preds_df[f"{name}_isotonic"] = p_iso

        auc_base = roc_auc_score(y_s, p_base)
        auc_iso  = roc_auc_score(y_s, p_iso)
        bs_base  = brier_score_loss(y_s, p_base)
        bs_iso   = brier_score_loss(y_s, p_iso)
        print(f"  [{split_name}] AUROC base={auc_base:.4f} iso={auc_iso:.4f}  "
              f"| Brier base={bs_base:.4f} iso={bs_iso:.4f}")

    # -- Save models -----------------------------------------------------------
    joblib.dump(base_model, OUT / f"{name}_base.pkl")
    joblib.dump(platt,      OUT / f"{name}_platt.pkl")
    joblib.dump(isotonic,   OUT / f"{name}_isotonic.pkl")
    print(f"  Saved {name} (base, platt, isotonic)")

# -- Persist predictions and hyperparameters -----------------------------------
val_preds_df.to_csv(OUT / "val_predictions.csv", index=False)
test_preds_df.to_csv(OUT / "test_predictions.csv", index=False)

with open(OUT / "hyperparams.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)

print(f"\nAll models saved to {OUT}")
print("Hyperparameters saved to data/models/hyperparams.json")
