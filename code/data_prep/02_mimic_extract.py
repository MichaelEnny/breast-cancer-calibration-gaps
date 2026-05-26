"""Extract breast cancer cohort from MIMIC-IV ED module and harmonise to Wisconsin feature space.

Data sources (data/mimic/ed/):
    diagnosis.csv.gz  -- ICD codes  (C50.x = breast cancer)
    edstays.csv.gz    -- demographics: race, gender, stay metadata
    triage.csv.gz     -- proxy vitals: HR, BP, SpO2, RR, temp, pain, acuity

Label definition:
    Positive (malignant = 1): patients with any ICD-10 C50.x code in their ED visit
    Negative (benign   = 0): patients with no malignancy/cancer code, randomly sampled
                              to create a balanced stress-test cohort

Feature harmonisation (30 Wisconsin slots):
    Wisconsin has 30 FNA nucleus features -- none exist in MIMIC-IV ED.
    We map 9 triage/demographic features to 9 Wisconsin positions (documented below).
    The remaining 21 positions are filled with Wisconsin training-set medians.
    This intentional mismatch IS the distributional stress-test (mirrors Paper 1 design).

    Feature mapping (index in Wisconsin feature list -> MIMIC source):
        0  mean_radius              <- heartrate   (rescaled)
        1  mean_texture             <- temperature (rescaled)
        2  mean_perimeter           <- sbp         (rescaled)
        3  mean_area                <- dbp         (rescaled)
        4  mean_smoothness          <- o2sat       (rescaled)
        5  mean_compactness         <- resprate    (rescaled)
        6  mean_concavity           <- pain        (rescaled, 0-10)
        7  mean_concave_points      <- acuity      (rescaled, 1-5)
        8  mean_symmetry            <- gender_enc  (0=F, 1=M)
        9-29                        <- Wisconsin training-set medians

Outputs (data/mimic/):
    bc_cohort_demographics.csv  -- subject_id, stay_id, race, gender, label, age_proxy
    bc_cohort_features.csv      -- 30-column harmonised feature matrix (Wisconsin space)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT    = Path(__file__).resolve().parents[2]
ED_DIR  = ROOT / "data" / "mimic" / "ed"
WIS_DIR = ROOT / "data" / "wisconsin"
OUT     = ROOT / "data" / "mimic"
OUT.mkdir(parents=True, exist_ok=True)

# -- 0. Check prerequisites ----------------------------------------------------
for req in ["X_train.csv", "train_stats.csv", "scaler.pkl", "feature_names.txt"]:
    if not (WIS_DIR / req).exists():
        raise FileNotFoundError(
            f"Missing {WIS_DIR / req}. Run 01_wisconsin_download.py first."
        )

feature_names = (WIS_DIR / "feature_names.txt").read_text().splitlines()
train_stats   = pd.read_csv(WIS_DIR / "train_stats.csv").set_index("feature")
scaler        = joblib.load(WIS_DIR / "scaler.pkl")
n_features    = len(feature_names)
print(f"Wisconsin features loaded: {n_features}")

# -- 1. Load MIMIC-IV ED tables ------------------------------------------------
print("Loading MIMIC-IV ED tables …")
diagnosis  = pd.read_csv(ED_DIR / "diagnosis.csv.gz")
edstays    = pd.read_csv(ED_DIR / "edstays.csv.gz")
triage     = pd.read_csv(ED_DIR / "triage.csv.gz")

print(f"  diagnosis : {len(diagnosis):,} rows")
print(f"  edstays   : {len(edstays):,} rows")
print(f"  triage    : {len(triage):,} rows")

# -- 2. Identify breast cancer patients (ICD-10 C50.x) ------------------------
# Filter to ICD-10 version only (version == 10)
bc_dx = diagnosis[
    (diagnosis["icd_version"] == 10) &
    (diagnosis["icd_code"].str.startswith("C50", na=False))
]
bc_stay_ids = bc_dx["stay_id"].unique()
print(f"\nBreast cancer stays (ICD-10 C50.x): {len(bc_stay_ids):,}")

# Also capture ICD-9 codes for breast cancer (174.x / 175.x) as fallback
bc_dx9 = diagnosis[
    (diagnosis["icd_version"] == 9) &
    (diagnosis["icd_code"].str.match(r"^17[45]", na=False))
]
bc_stay_ids_9 = bc_dx9["stay_id"].unique()
all_bc_stays = np.union1d(bc_stay_ids, bc_stay_ids_9)
print(f"Breast cancer stays (ICD-10 + ICD-9):  {len(all_bc_stays):,}")

# -- 3. Build positive cohort --------------------------------------------------
pos_stays = edstays[edstays["stay_id"].isin(all_bc_stays)].copy()
pos_stays["label"] = 1
print(f"\nPositive cohort (breast cancer):  {len(pos_stays):,} unique stays")

# -- 4. Build negative cohort (no cancer diagnosis at all) --------------------
# Exclude any ICD codes starting with C or D (neoplasms), 140-239 ICD-9 (neoplasms)
cancer_icd10 = diagnosis[
    (diagnosis["icd_version"] == 10) &
    (diagnosis["icd_code"].str.match(r"^[CD]", na=False))
]["stay_id"].unique()

cancer_icd9 = diagnosis[
    (diagnosis["icd_version"] == 9) &
    (diagnosis["icd_code"].str.match(r"^1[4-9]\d|^2[0-3]\d", na=False))
]["stay_id"].unique()

cancer_stays = np.union1d(cancer_icd10, cancer_icd9)
neg_pool = edstays[~edstays["stay_id"].isin(cancer_stays)].copy()

# Sample negatives: up to 3× positive count (or all if fewer available)
n_pos = len(pos_stays)
n_neg = min(len(neg_pool), n_pos * 3)
rng   = np.random.default_rng(42)
neg_idx = rng.choice(len(neg_pool), size=n_neg, replace=False)
neg_stays = neg_pool.iloc[neg_idx].copy()
neg_stays["label"] = 0
print(f"Negative cohort (non-cancer controls): {len(neg_stays):,} stays")

# -- 5. Combine cohort and merge triage features -------------------------------
cohort = pd.concat([pos_stays, neg_stays], ignore_index=True)
cohort = cohort.merge(
    triage[["stay_id", "temperature", "heartrate", "resprate",
            "o2sat", "sbp", "dbp", "pain", "acuity"]],
    on="stay_id", how="left"
)

# -- 6. Clean demographics -----------------------------------------------------
cohort["gender"] = cohort["gender"].str.upper().str.strip()
cohort["race"]   = cohort["race"].str.upper().str.strip()

# Encode gender: F->0, M->1, unknown->NaN
gender_map = {"F": 0, "M": 1, "FEMALE": 0, "MALE": 1}
cohort["gender_enc"] = cohort["gender"].map(gender_map)

# Standardise race categories
def standardise_race(r):
    if pd.isna(r):
        return "UNKNOWN"
    r = str(r).upper()
    if "WHITE"    in r:               return "WHITE"
    if "BLACK"    in r or "AFRICAN" in r: return "BLACK"
    if "HISPANIC" in r or "LATINO" in r:  return "HISPANIC"
    if "ASIAN"    in r:               return "ASIAN"
    if "UNKNOWN"  in r or "UNABLE" in r or "OTHER" in r: return "OTHER"
    return "OTHER"

cohort["race_std"] = cohort["race"].apply(standardise_race)

print("\nRace distribution:")
print(cohort["race_std"].value_counts())
print(f"\nGender distribution:")
print(cohort["gender"].value_counts())
print(f"\nLabel distribution:")
print(cohort["label"].value_counts())

# -- 7. Save demographics ------------------------------------------------------
demog_cols = ["subject_id", "stay_id", "race_std", "gender", "gender_enc",
              "label", "intime"]
cohort[demog_cols].to_csv(OUT / "bc_cohort_demographics.csv", index=False)
print(f"\nDemographics saved: {len(cohort)} rows")

# -- 8. Build 30-feature harmonised feature matrix ----------------------------
#
# Feature mapping  (9 MIMIC triage/demographic -> 9 Wisconsin positions):
#   Wisconsin idx | Wisconsin feature name | MIMIC source
#   -------------|------------------------|---------------------------
#   0            | mean_radius            | heartrate   (normal ~60-100 bpm)
#   1            | mean_texture           | temperature (normal ~97-99 F)
#   2            | mean_perimeter         | sbp         (normal ~100-140 mmHg)
#   3            | mean_area              | dbp         (normal ~60-90 mmHg)
#   4            | mean_smoothness        | o2sat       (normal ~95-100 %)
#   5            | mean_compactness       | resprate    (normal ~12-20 /min)
#   6            | mean_concavity         | pain        (0-10 scale)
#   7            | mean_concave_points    | acuity      (1-5 scale)
#   8            | mean_symmetry          | gender_enc  (0 or 1)
#   9-29         | SE and worst features  | Wisconsin training-set medians
#
# All values are later passed through the Wisconsin StandardScaler.
# The result is NOT a meaningful FNA feature vector -- it is deliberately
# out-of-distribution, which is the whole point of the stress test.

MAPPED = {
    0: "heartrate",
    1: "temperature",
    2: "sbp",
    3: "dbp",
    4: "o2sat",
    5: "resprate",
    6: "pain",
    7: "acuity",
    8: "gender_enc",
}

# Start with training-set medians for all 30 features
medians = train_stats["median"].values   # shape (30,)
feat_matrix = np.tile(medians, (len(cohort), 1)).astype(float)  # (n, 30)

for col_idx, mimic_col in MAPPED.items():
    # Use pd.to_numeric so non-numeric strings (e.g. "severe" in pain) become NaN
    vals = pd.to_numeric(cohort[mimic_col], errors="coerce").values.astype(float)
    # Replace NaN with Wisconsin training median
    nan_mask = np.isnan(vals)
    vals[nan_mask] = medians[col_idx]
    feat_matrix[:, col_idx] = vals

# Apply Wisconsin StandardScaler
feat_scaled = scaler.transform(feat_matrix)

feat_df = pd.DataFrame(feat_scaled, columns=feature_names)
feat_df.insert(0, "stay_id", cohort["stay_id"].values)
feat_df.to_csv(OUT / "bc_cohort_features.csv", index=False)

print(f"Feature matrix saved: {feat_df.shape}  ->  {OUT / 'bc_cohort_features.csv'}")
print("\nDone. MIMIC-IV ED breast cancer cohort ready.")
print(f"  Positives (C50.x):      {cohort['label'].sum():,}")
print(f"  Negatives (controls):   {(cohort['label']==0).sum():,}")
print(f"  Race groups available:  {cohort['race_std'].nunique()}")
