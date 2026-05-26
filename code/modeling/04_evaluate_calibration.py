"""Evaluate calibration and discrimination for all 5 models × 3 variants.

Wisconsin test set  -> tables/T1_wisconsin_calibration.csv
MIMIC-IV cohort     -> tables/T2_mimic_calibration.csv
                      (also fits + evaluates subgroup-targeted Platt scaling)
MIMIC predictions   -> data/models/mimic_predictions.csv  (used by 05_dcgs_analysis.py)

Metrics per row:
    ECE      Expected Calibration Error (10 equal-width bins)
    MCE      Maximum Calibration Error (max bin |acc − conf|)
    Brier    Brier Score
    BSS      Brier Skill Score  = 1 − BS / BS_ref  (BS_ref = prevalence × (1−prevalence))
    AUROC    Area Under ROC
    AUPRC    Area Under Precision-Recall Curve
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "models"
WIS_DIR   = ROOT / "data" / "wisconsin"
MIM_DIR   = ROOT / "data" / "mimic"
OUT       = ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# -- Import shared metric helpers ----------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # adds code/ to path
from dcgs.metric import compute_ece

NAMES    = ["LR", "RF", "SVM", "NB", "XGB"]
VARIANTS = ["base", "platt", "isotonic"]


# -- Helper metrics ------------------------------------------------------------

def compute_mce(y_true, y_prob, n_bins=10):
    """Maximum Calibration Error -- worst-bin |accuracy − confidence|."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(y_true[mask].mean() - y_prob[mask].mean()))
    return mce


def brier_skill_score(y_true, y_prob):
    """BSS = 1 − BS / BS_ref, where BS_ref is climatological (prevalence) baseline."""
    bs     = brier_score_loss(y_true, y_prob)
    p_bar  = y_true.mean()
    bs_ref = p_bar * (1 - p_bar)
    if bs_ref == 0:
        return 0.0
    return 1.0 - bs / bs_ref


def evaluate_row(name, variant, y_true, y_prob):
    """Return a dict of all metrics for one model-variant on one split."""
    return {
        "model":   name,
        "variant": variant,
        "n":       len(y_true),
        "prev":    float(y_true.mean()),
        "ECE":     round(compute_ece(y_true, y_prob), 4),
        "MCE":     round(compute_mce(y_true, y_prob), 4),
        "Brier":   round(brier_score_loss(y_true, y_prob), 4),
        "BSS":     round(brier_skill_score(y_true, y_prob), 4),
        "AUROC":   round(roc_auc_score(y_true, y_prob), 4),
        "AUPRC":   round(average_precision_score(y_true, y_prob), 4),
    }


# -- Load Wisconsin splits -----------------------------------------------------
print("Loading Wisconsin data …")
X_val  = pd.read_csv(WIS_DIR / "X_val.csv").values
y_val  = pd.read_csv(WIS_DIR / "y_val.csv").values.ravel()
X_test = pd.read_csv(WIS_DIR / "X_test.csv").values
y_test = pd.read_csv(WIS_DIR / "y_test.csv").values.ravel()
print(f"  Val={len(X_val)}  Test={len(X_test)}")

# -- Load MIMIC cohort ---------------------------------------------------------
print("Loading MIMIC-IV cohort …")
demog_path = MIM_DIR / "bc_cohort_demographics.csv"
feat_path  = MIM_DIR / "bc_cohort_features.csv"

if not demog_path.exists() or not feat_path.exists():
    raise FileNotFoundError(
        "MIMIC cohort files not found. Run 02_mimic_extract.py first.\n"
        f"  Expected: {demog_path}\n           {feat_path}"
    )

demog = pd.read_csv(demog_path)
feats = pd.read_csv(feat_path)

# Align on stay_id
cohort = demog.merge(feats, on="stay_id", how="inner")

feature_names = list(pd.read_csv(WIS_DIR / "X_train.csv").columns)
X_mimic = cohort[feature_names].values
y_mimic = cohort["label"].values
race_mimic   = cohort["race_std"].values
gender_mimic = cohort["gender"].values

print(f"  MIMIC cohort: n={len(cohort)}  prevalence={y_mimic.mean():.3f}")

# 50/50 split within MIMIC for subgroup Platt calibration
idx_cal, idx_test_m = train_test_split(
    np.arange(len(cohort)), test_size=0.50, random_state=42,
    stratify=y_mimic
)
X_mcal, y_mcal = X_mimic[idx_cal],    y_mimic[idx_cal]
X_mtest, y_mtest = X_mimic[idx_test_m], y_mimic[idx_test_m]
race_mtest = race_mimic[idx_test_m]
gender_mtest = gender_mimic[idx_test_m]

print(f"  MIMIC cal={len(X_mcal)}  MIMIC eval={len(X_mtest)}")

# -- Evaluate on Wisconsin test set -> T1 ---------------------------------------
print("\nEvaluating on Wisconsin test set …")
t1_rows = []

for name in NAMES:
    for variant in VARIANTS:
        mdl = joblib.load(MODEL_DIR / f"{name}_{variant}.pkl")
        y_prob = mdl.predict_proba(X_test)[:, 1]
        t1_rows.append(evaluate_row(name, variant, y_test, y_prob))
        print(f"  {name:4s} {variant:8s}  ECE={t1_rows[-1]['ECE']:.4f}  "
              f"AUROC={t1_rows[-1]['AUROC']:.4f}")

T1 = pd.DataFrame(t1_rows)
T1.to_csv(OUT / "T1_wisconsin_calibration.csv", index=False)
print(f"\nT1 saved -> {OUT / 'T1_wisconsin_calibration.csv'}")

# -- Evaluate on MIMIC test portion -> T2 ---------------------------------------
print("\nEvaluating on MIMIC-IV cohort …")
t2_rows = []

# Track predictions for every MIMIC eval row (for 05_dcgs_analysis.py)
mimic_pred_df = pd.DataFrame({
    "stay_id":  cohort["stay_id"].values[idx_test_m],
    "y_true":   y_mtest,
    "race_std": race_mtest,
    "gender":   gender_mtest,
})

for name in NAMES:
    for variant in VARIANTS:
        mdl    = joblib.load(MODEL_DIR / f"{name}_{variant}.pkl")
        y_prob = mdl.predict_proba(X_mtest)[:, 1]
        t2_rows.append(evaluate_row(name, variant, y_mtest, y_prob))
        mimic_pred_df[f"{name}_{variant}"] = y_prob
        print(f"  {name:4s} {variant:8s}  ECE={t2_rows[-1]['ECE']:.4f}  "
              f"AUROC={t2_rows[-1]['AUROC']:.4f}")

    # -- Subgroup-targeted Platt scaling (fitted on MIMIC cal half) ------------
    base_mdl = joblib.load(MODEL_DIR / f"{name}_base.pkl")

    race_groups = np.unique(race_mimic[idx_cal])
    sg_preds    = np.full(len(X_mtest), np.nan)

    for g in race_groups:
        g_cal_mask  = race_mimic[idx_cal]  == g
        g_test_mask = race_mtest            == g
        if g_cal_mask.sum() < 30 or g_test_mask.sum() == 0:
            # Fall back to global platt for tiny subgroups
            global_platt = joblib.load(MODEL_DIR / f"{name}_platt.pkl")
            sg_preds[g_test_mask] = global_platt.predict_proba(
                X_mtest[g_test_mask]
            )[:, 1]
            continue
        sg_platt = CalibratedClassifierCV(base_mdl, method="sigmoid", cv=None, ensemble=False)
        sg_platt.fit(X_mcal[g_cal_mask], y_mcal[g_cal_mask])
        sg_preds[g_test_mask] = sg_platt.predict_proba(
            X_mtest[g_test_mask]
        )[:, 1]

    # Fill any remaining NaNs (e.g. unseen race groups) with global platt
    nan_mask = np.isnan(sg_preds)
    if nan_mask.any():
        global_platt = joblib.load(MODEL_DIR / f"{name}_platt.pkl")
        sg_preds[nan_mask] = global_platt.predict_proba(X_mtest[nan_mask])[:, 1]

    row_sg = evaluate_row(name, "subgroup_platt", y_mtest, sg_preds)
    t2_rows.append(row_sg)
    mimic_pred_df[f"{name}_subgroup_platt"] = sg_preds
    print(f"  {name:4s} subgroup_platt  ECE={row_sg['ECE']:.4f}  "
          f"AUROC={row_sg['AUROC']:.4f}")

T2 = pd.DataFrame(t2_rows)
T2.to_csv(OUT / "T2_mimic_calibration.csv", index=False)
print(f"\nT2 saved -> {OUT / 'T2_mimic_calibration.csv'}")

mimic_pred_df.to_csv(MODEL_DIR / "mimic_predictions.csv", index=False)
print(f"MIMIC predictions saved -> {MODEL_DIR / 'mimic_predictions.csv'}")

print("\n04_evaluate_calibration.py complete.")
