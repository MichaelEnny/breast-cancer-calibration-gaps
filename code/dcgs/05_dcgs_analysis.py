"""Compute DCGS by subgroup axis (race, gender) for all model × calibration variants.

Reads:
    data/models/mimic_predictions.csv   -- predicted probabilities on MIMIC eval set
                                          (written by 04_evaluate_calibration.py)

Outputs:
    tables/T3_dcgs_results.csv          -- DCGS range/spread, bootstrap CI, per-group ECE
    tables/T3_pairwise_tests.csv        -- Bonferroni-corrected pairwise Wilcoxon results
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
from dcgs.metric import compute_dcgs, bootstrap_dcgs_ci, pairwise_ece_tests

MODEL_DIR = ROOT / "data" / "models"
OUT       = ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)

NAMES    = ["LR", "RF", "SVM", "NB", "XGB"]
VARIANTS = ["base", "platt", "isotonic", "subgroup_platt"]

# -- Load MIMIC predictions ----------------------------------------------------
pred_path = MODEL_DIR / "mimic_predictions.csv"
if not pred_path.exists():
    raise FileNotFoundError(
        f"{pred_path} not found. Run 04_evaluate_calibration.py first."
    )

preds = pd.read_csv(pred_path)
y_true       = preds["y_true"].values
race_labels  = preds["race_std"].values
gender_labels = preds["gender"].values

print(f"MIMIC eval set: n={len(preds)}")
print(f"  Race groups  : {np.unique(race_labels)}")
print(f"  Gender groups: {np.unique(gender_labels)}")

SUBGROUP_AXES = {
    "race":   race_labels,
    "gender": gender_labels,
}

# -- Compute DCGS for every model × variant × subgroup axis -------------------
t3_rows      = []
pairwise_all = []

for name in NAMES:
    for variant in VARIANTS:
        col = f"{name}_{variant}"
        if col not in preds.columns:
            continue
        y_prob = preds[col].values

        for axis_name, sg_labels in SUBGROUP_AXES.items():
            sg_arr = np.array(sg_labels)
            ece_per_group, dcgs_r, dcgs_s = compute_dcgs(
                y_true, y_prob, sg_arr, min_n=30
            )

            # Bootstrap CI on DCGS range (skip if fewer than 2 qualifying groups)
            ci_lo, ci_hi = (None, None)
            if dcgs_r is not None:
                ci_lo, ci_hi = bootstrap_dcgs_ci(
                    y_true, y_prob, sg_arr,
                    n_bootstrap=1000, alpha=0.05, min_n=30
                )

            row = {
                "model":          name,
                "variant":        variant,
                "subgroup_axis":  axis_name,
                "n_groups":       len(ece_per_group),
                "dcgs_range":     round(dcgs_r, 4)  if dcgs_r  is not None else None,
                "dcgs_spread":    round(dcgs_s, 4)  if dcgs_s  is not None else None,
                "ci_lo_95":       round(ci_lo, 4)   if ci_lo   is not None else None,
                "ci_hi_95":       round(ci_hi, 4)   if ci_hi   is not None else None,
                "above_threshold": (dcgs_r is not None and dcgs_r > 0.05),
            }
            # Embed per-group ECE as separate columns
            for g, ece_g in sorted(ece_per_group.items()):
                row[f"ece_{g}"] = round(ece_g, 4)

            t3_rows.append(row)

            print(f"  {name:4s} {variant:14s} [{axis_name:6s}]  "
                  f"DCGS_range={dcgs_r:.4f}" if dcgs_r is not None
                  else f"  {name:4s} {variant:14s} [{axis_name:6s}]  "
                       f"DCGS_range=N/A (< 2 qualifying groups)")

            # Pairwise Wilcoxon tests
            pw = pairwise_ece_tests(y_true, y_prob, sg_arr, min_n=30)
            for pw_row in pw:
                pw_row["model"]         = name
                pw_row["variant"]       = variant
                pw_row["subgroup_axis"] = axis_name
                pairwise_all.append(pw_row)

T3 = pd.DataFrame(t3_rows)
T3.to_csv(OUT / "T3_dcgs_results.csv", index=False)
print(f"\nT3 saved -> {OUT / 'T3_dcgs_results.csv'}")

T3_pw = pd.DataFrame(pairwise_all) if pairwise_all else pd.DataFrame()
T3_pw.to_csv(OUT / "T3_pairwise_tests.csv", index=False)
print(f"T3 pairwise tests saved -> {OUT / 'T3_pairwise_tests.csv'}")

# -- Summary: models with DCGS > 0.05 threshold -------------------------------
print("\n-- DCGS > 0.05 threshold summary ----------------------")
flagged = T3[T3["above_threshold"] == True][["model", "variant", "subgroup_axis", "dcgs_range"]]
if len(flagged):
    print(flagged.to_string(index=False))
else:
    print("  No model-variant pairs exceed DCGS > 0.05 threshold.")

print("\n05_dcgs_analysis.py complete.")
