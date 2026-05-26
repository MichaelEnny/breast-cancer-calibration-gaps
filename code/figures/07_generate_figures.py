"""Generate all figures for Paper 2.

Figures produced (under second_paper/figures/):
    reliability_wisconsin/{MODEL}_reliability.png  -- Wisconsin test, all calibration variants
    reliability_mimic/{MODEL}_reliability.png      -- MIMIC cohort, all calibration variants
    reliability_mimic/{MODEL}_subgroup_race.png    -- MIMIC cohort by racial subgroup
    reliability_mimic/{MODEL}_subgroup_gender.png  -- MIMIC cohort by gender subgroup
    dcgs_heatmap_race.png                          -- DCGS range × model × variant (race axis)
    dcgs_heatmap_gender.png                        -- DCGS range × model × variant (gender axis)

Reads:
    data/models/test_predictions.csv
    data/models/mimic_predictions.csv
    data/wisconsin/y_test.csv
    tables/T3_dcgs_results.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(FIG_DIR / "reliability_wisconsin").mkdir(exist_ok=True)
(FIG_DIR / "reliability_mimic").mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
# Change working dir so reliability_diagrams.py's os.makedirs uses absolute paths
import os
os.chdir(ROOT)

from figures.reliability_diagrams import (
    plot_reliability, plot_subgroup_reliability, plot_dcgs_heatmap
)

MODEL_DIR = ROOT / "data" / "models"
WIS_DIR   = ROOT / "data" / "wisconsin"
TAB_DIR   = ROOT / "tables"

NAMES         = ["LR", "RF", "SVM", "NB", "XGB"]
VARIANT_LABELS = {
    "base":     "No calib",
    "platt":    "Platt",
    "isotonic": "Isotonic",
    "subgroup_platt": "Subgroup Platt",
}
MODEL_ORDER   = NAMES
VARIANT_ORDER = ["base", "platt", "isotonic", "subgroup_platt"]

# -- Load data -----------------------------------------------------------------
print("Loading prediction files …")

y_test   = pd.read_csv(WIS_DIR / "y_test.csv").values.ravel()
test_df  = pd.read_csv(MODEL_DIR / "test_predictions.csv")

mimic_path = MODEL_DIR / "mimic_predictions.csv"
if not mimic_path.exists():
    raise FileNotFoundError(
        f"{mimic_path} not found. Run 04_evaluate_calibration.py first."
    )
mimic_df = pd.read_csv(mimic_path)
y_mimic  = mimic_df["y_true"].values
race_arr = mimic_df["race_std"].values
gen_arr  = mimic_df["gender"].values

dcgs_path = TAB_DIR / "T3_dcgs_results.csv"
if not dcgs_path.exists():
    raise FileNotFoundError(
        f"{dcgs_path} not found. Run 05_dcgs_analysis.py first."
    )
dcgs_df = pd.read_csv(dcgs_path)

# -- 1. Wisconsin population reliability diagrams ------------------------------
print("\nGenerating Wisconsin reliability diagrams …")
for name in NAMES:
    prob_dict = {}
    for var, label in VARIANT_LABELS.items():
        col = f"{name}_{var}"
        if col in test_df.columns:
            prob_dict[label] = test_df[col].values
    if not prob_dict:
        continue
    save_path = FIG_DIR / "reliability_wisconsin" / f"{name}_reliability.png"
    plot_reliability(y_test, prob_dict,
                     title=f"{name} -- Wisconsin Diagnostic (n=114)",
                     save_path=str(save_path))

# -- 2. MIMIC population reliability diagrams ---------------------------------
print("\nGenerating MIMIC population reliability diagrams …")
for name in NAMES:
    prob_dict = {}
    for var, label in VARIANT_LABELS.items():
        col = f"{name}_{var}"
        if col in mimic_df.columns:
            prob_dict[label] = mimic_df[col].values
    if not prob_dict:
        continue
    save_path = FIG_DIR / "reliability_mimic" / f"{name}_reliability.png"
    plot_reliability(y_mimic, prob_dict,
                     title=f"{name} -- MIMIC-IV Breast Cancer Cohort",
                     save_path=str(save_path))

# -- 3. MIMIC subgroup reliability diagrams (race + gender) --------------------
print("\nGenerating MIMIC subgroup reliability diagrams …")
for name in NAMES:
    # Use the best variant available: prefer subgroup_platt -> isotonic -> platt -> base
    y_prob = None
    for var in ["subgroup_platt", "isotonic", "platt", "base"]:
        col = f"{name}_{var}"
        if col in mimic_df.columns:
            y_prob = mimic_df[col].values
            chosen_var = var
            break
    if y_prob is None:
        continue

    # Race subgroups
    save_path = FIG_DIR / "reliability_mimic" / f"{name}_subgroup_race.png"
    plot_subgroup_reliability(
        y_mimic, y_prob, race_arr,
        title=f"{name} ({chosen_var}) -- MIMIC by Race",
        save_path=str(save_path)
    )

    # Gender subgroups
    save_path = FIG_DIR / "reliability_mimic" / f"{name}_subgroup_gender.png"
    plot_subgroup_reliability(
        y_mimic, y_prob, gen_arr,
        title=f"{name} ({chosen_var}) -- MIMIC by Gender",
        save_path=str(save_path)
    )

# -- 4. DCGS heatmaps ---------------------------------------------------------
print("\nGenerating DCGS heatmaps …")

for axis in ["race", "gender"]:
    subset = dcgs_df[dcgs_df["subgroup_axis"] == axis]
    if subset.empty:
        print(f"  No DCGS data for axis={axis}, skipping heatmap.")
        continue

    for val_col in ["dcgs_range", "dcgs_spread"]:
        save_path = FIG_DIR / f"dcgs_heatmap_{axis}_{val_col}.png"
        plot_dcgs_heatmap(
            subset, val_col,
            title=f"DCGS ({val_col.replace('_', ' ')}) by Race: {axis}",
            save_path=str(save_path),
            model_order=MODEL_ORDER,
            variant_order=VARIANT_ORDER,
        )

print("\n07_generate_figures.py complete.")
print(f"All figures saved under: {FIG_DIR}")
