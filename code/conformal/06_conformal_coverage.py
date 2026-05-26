"""Split conformal prediction -- Wisconsin + MIMIC subgroup coverage.

Procedure:
  1. Use Wisconsin val predictions as the conformal calibration set
  2. Compute threshold at 90% marginal coverage (alpha=0.10) per model × variant
  3. Apply threshold to Wisconsin test set -- report marginal coverage
  4. Apply to MIMIC eval set -- report per-subgroup coverage and DCG

Reads:
    data/models/val_predictions.csv       -- Wisconsin val predicted probs (from 03)
    data/models/test_predictions.csv      -- Wisconsin test predicted probs (from 03)
    data/models/mimic_predictions.csv     -- MIMIC eval predicted probs (from 04)
    data/wisconsin/y_val.csv, y_test.csv

Outputs:
    tables/T4_conformal_coverage.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
from conformal.subgroup_coverage import (
    nonconformity_scores, split_conformal_threshold,
    prediction_sets, subgroup_coverage
)

MODEL_DIR = ROOT / "data" / "models"
WIS_DIR   = ROOT / "data" / "wisconsin"
OUT       = ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10   # target 90% marginal coverage
NAMES    = ["LR", "RF", "SVM", "NB", "XGB"]
VARIANTS = ["base", "platt", "isotonic", "subgroup_platt"]

# -- Load data -----------------------------------------------------------------
print("Loading predictions …")
y_val  = pd.read_csv(WIS_DIR  / "y_val.csv").values.ravel()
y_test = pd.read_csv(WIS_DIR  / "y_test.csv").values.ravel()

val_preds  = pd.read_csv(MODEL_DIR / "val_predictions.csv")
test_preds = pd.read_csv(MODEL_DIR / "test_predictions.csv")

mimic_path = MODEL_DIR / "mimic_predictions.csv"
if not mimic_path.exists():
    raise FileNotFoundError(
        f"{mimic_path} not found. Run 04_evaluate_calibration.py first."
    )
mimic_preds = pd.read_csv(mimic_path)
y_mimic      = mimic_preds["y_true"].values
race_mimic   = mimic_preds["race_std"].values
gender_mimic = mimic_preds["gender"].values

print(f"  Val={len(y_val)}  Test={len(y_test)}  MIMIC={len(y_mimic)}")

# -- Evaluate per model × variant ----------------------------------------------
rows = []

for name in NAMES:
    for variant in VARIANTS:
        col = f"{name}_{variant}"

        # Wisconsin val column: variant names are base/platt/isotonic (no subgroup)
        # subgroup_platt is MIMIC-only -- fall back to platt for Wisconsin conformal cal
        wis_col = col if col in val_preds.columns else f"{name}_platt"
        if wis_col not in val_preds.columns:
            continue

        # -- Conformal threshold from Wisconsin val ----------------------------
        p_val       = val_preds[wis_col].values
        nc_val      = nonconformity_scores(p_val)
        threshold   = split_conformal_threshold(nc_val, alpha=ALPHA)

        # -- Wisconsin test coverage -------------------------------------------
        if col in test_preds.columns:
            p_test_wis = test_preds[col].values
        else:
            p_test_wis = test_preds[f"{name}_platt"].values

        pred_set_wis = prediction_sets(p_test_wis, threshold)
        wis_coverage = float((pred_set_wis == y_test).mean())
        wis_set_size = float(pred_set_wis.mean())  # fraction included

        # -- MIMIC coverage (marginal + subgroup) ------------------------------
        if col not in mimic_preds.columns:
            continue

        p_mimic   = mimic_preds[col].values
        pred_set_m = prediction_sets(p_mimic, threshold)
        mim_cov    = float((pred_set_m == y_mimic).mean())
        mim_size   = float(pred_set_m.mean())

        # Per-subgroup coverage + DCG -- race axis
        cov_race, dcg_race = subgroup_coverage(
            y_mimic, pred_set_m, race_mimic, min_n=30
        )
        # Per-subgroup coverage + DCG -- gender axis
        cov_gender, dcg_gender = subgroup_coverage(
            y_mimic, pred_set_m, gender_mimic, min_n=30
        )

        row = {
            "model":            name,
            "variant":          variant,
            "threshold":        round(threshold, 4),
            "wis_coverage":     round(wis_coverage, 4),
            "wis_set_size":     round(wis_set_size, 4),
            "mimic_coverage":   round(mim_cov, 4),
            "mimic_set_size":   round(mim_size, 4),
            "dcg_race":         round(dcg_race,   4) if dcg_race   is not None else None,
            "dcg_gender":       round(dcg_gender, 4) if dcg_gender is not None else None,
        }

        # Embed per-race and per-gender coverage
        for g, cov_g in sorted(cov_race.items()):
            row[f"cov_race_{g}"] = round(cov_g, 4)
        for g, cov_g in sorted(cov_gender.items()):
            row[f"cov_gender_{g}"] = round(cov_g, 4)

        rows.append(row)

        print(f"  {name:4s} {variant:14s}  thr={threshold:.3f}  "
              f"wis_cov={wis_coverage:.3f}  mim_cov={mim_cov:.3f}  "
              f"DCG_race={dcg_race:.3f}" if dcg_race is not None
              else f"  {name:4s} {variant:14s}  thr={threshold:.3f}  "
                   f"wis_cov={wis_coverage:.3f}  mim_cov={mim_cov:.3f}")

T4 = pd.DataFrame(rows)
T4.to_csv(OUT / "T4_conformal_coverage.csv", index=False)
print(f"\nT4 saved -> {OUT / 'T4_conformal_coverage.csv'}")

# -- Summary: DCG > 0.05 threshold ---------------------------------------------
print("\n-- DCG > 0.05 threshold summary (race axis) --------------")
flagged = T4[T4["dcg_race"].notna() & (T4["dcg_race"] > 0.05)][
    ["model", "variant", "mimic_coverage", "dcg_race"]
]
if len(flagged):
    print(flagged.to_string(index=False))
else:
    print("  No model-variant pairs exceed DCG > 0.05 on race axis.")

print("\n06_conformal_coverage.py complete.")
