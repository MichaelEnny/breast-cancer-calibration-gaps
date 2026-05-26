# Paper 2 — Step-by-Step Completion Guide

**Title:** Demographic Calibration Gaps in Breast Cancer Risk Prediction: Introducing the DCGS Metric Using Wisconsin Diagnostic and MIMIC-IV

**Type:** Calibration equity + fairness evaluation paper
**Target Venue:** *JAMIA Open* (primary); *Journal of Biomedical Informatics* (backup)
**Target Submission:** End of Year 1
**Open-Source Requirement:** DCGS computation library + evaluation pipeline, Apache 2.0, GitHub

---

## What This Paper Must Prove

This paper extends the calibration evaluation work of Paper 1 (CKD) to breast cancer risk prediction and adds a critical new dimension: **calibration equity across demographic subgroups**.

Paper 1 showed that calibration collapses under external validation. Paper 2 shows that even when a model validates well internally, calibration can be systematically worse for specific demographic groups when tested on an external population — and that this disparity is invisible to researchers who only report population-level ECE.

The contribution is the **Demographic Calibration Gap Score (DCGS)**: a metric that quantifies how much calibration varies across demographic subgroups, making disparities visible, comparable, and actionable.

**Study design:**
- **Primary cohort:** Wisconsin Diagnostic Breast Cancer (UCI, n=569) — model training, internal calibration evaluation, conformal prediction
- **External validation cohort:** MIMIC-IV breast cancer patients (ICD-10 C50.x filter) — distributional stress-test + demographic DCGS analysis (race, gender, age, insurance)

This two-dataset design directly mirrors Paper 1's UCI → MIMIC-IV structure, creating a coherent methodological thread across Papers 1 and 2.

---

## Gap Statement (Use This in Your Introduction)

- Only **3.2% of breast cancer ML studies assessed calibration** (PMC systematic scoping review of XAI in breast cancer, 2024)
- **75–100% of studies do not report race/ethnicity**; most training data is from White patients (same review)
- "More than half of included models rated high risk of bias due to insufficient calibration reporting" (BMC Cancer systematic review of 107 prediction models, 2025)
- No published breast cancer prediction study has introduced a **demographic-stratified calibration metric** — existing fairness metrics measure discrimination (AUROC parity) not calibration equity
- Single-center external validation datasets like MIMIC-IV reveal how calibration shifts under demographic and distributional change — but no study has used MIMIC-IV to measure demographic calibration equity specifically in breast cancer prediction

---

## The New Metric: Demographic Calibration Gap Score (DCGS)

**Definition:** DCGS quantifies the spread of calibration error across demographic subgroups.

```
For K subgroups (defined by race, age tier, or race × age tier):
  ECE_k  =  Expected Calibration Error for subgroup k (equal-width binning, 10 bins)

DCGS  =  max(ECE_k) − min(ECE_k)     [range version — simple, interpretable]
        OR
DCGS  =  std(ECE_k)                   [spread version — sensitive to intermediate gaps]
```

**Which version to use:** Report both. Use the range version as your headline metric (most interpretable for clinicians) and the spread version for statistical testing.

**Threshold:** DCGS > 0.05 (range) indicates clinically meaningful demographic calibration disparity — justifying subgroup-targeted recalibration rather than global recalibration.

**Why this is novel:**
- ECE and Brier Score are population-level metrics; no published metric specifically quantifies *between-subgroup* calibration variation
- DCGS is complementary to fairness metrics like equalized odds (which measure discrimination parity) — DCGS measures *probability accuracy parity*
- A model can have perfect equalized odds and high DCGS simultaneously — meaning predicted probabilities are equally wrong in different directions for different groups

---

## Section 1: Primary Dataset — Wisconsin Diagnostic Breast Cancer (UCI)

### 1.1 Dataset Overview

- **Source:** UCI Machine Learning Repository — free, no account or credentialing required
- **URL:** archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Size:** n = 569 patients (357 benign, 212 malignant)
- **Features:** 30 numeric features computed from FNA (fine needle aspirate) cell nuclei images — radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension (mean, SE, worst for each)
- **Outcome:** Binary — malignant (1) vs. benign (0)
- **Limitation:** No demographic variables (race, gender, age) — DCGS cannot be computed on this dataset alone; it is used for model training and internal calibration evaluation only

### 1.2 Download Steps

```python
# Option 1: Direct download via ucimlrepo
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Option 2: Manual download
# Go to: archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# Download wdbc.data and wdbc.names
# Place in second_paper/data/wisconsin/
```

### 1.3 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Train: n=455 | Test: n=114
```

### 1.4 Features and Preprocessing

- Scale all 30 features using `StandardScaler` (fit on train, apply to test and MIMIC-IV)
- No missing values in Wisconsin Diagnostic — no imputation needed
- Save the fitted scaler for application to MIMIC-IV features during external validation

---

## Section 2: External Validation Dataset — MIMIC-IV Breast Cancer Cohort

### 2.1 Why MIMIC-IV for External Validation

- MIMIC-IV is the same dataset used in Paper 1 (CKD) — creates methodological consistency
- Contains `race`, `gender`, `anchor_age`, and `insurance` columns — sufficient for DCGS computation
- Breast cancer patients can be extracted via ICD-10 C50.x diagnosis codes
- PhysioNet access is already being obtained for Papers 7 and 9 — no separate application needed
- Single-center (Beth Israel Deaconess Medical Center, Boston) — an honest limitation, explicitly stated

### 2.2 MIMIC-IV Access

You need **MIMIC-IV full release** (not the demo) for this paper.
- Apply at: physionet.org/content/mimiciv/
- Same CITI certificate used for MIMIC-IV-ED applies
- Expected approval: a few days to 2 weeks

### 2.3 Cohort Extraction

Extract breast cancer patients from MIMIC-IV using ICD-10 codes:

```python
import pandas as pd

# Load MIMIC-IV tables (after download)
diagnoses = pd.read_csv("mimic-iv/hosp/diagnoses_icd.csv.gz")
patients  = pd.read_csv("mimic-iv/hosp/patients.csv.gz")
admissions = pd.read_csv("mimic-iv/hosp/admissions.csv.gz")

# Filter breast cancer diagnoses (ICD-10 C50.x)
bc_dx = diagnoses[diagnoses['icd_code'].str.startswith('C50')]
bc_subjects = bc_dx['subject_id'].unique()

# Merge with patient demographics
patients_bc = patients[patients['subject_id'].isin(bc_subjects)].copy()
admissions_bc = admissions[admissions['subject_id'].isin(bc_subjects)].copy()

cohort = patients_bc.merge(
    admissions_bc[['subject_id', 'race', 'insurance']].drop_duplicates('subject_id'),
    on='subject_id', how='left'
)

# Derive age tier from anchor_age
cohort['age_tier'] = pd.cut(
    cohort['anchor_age'],
    bins=[0, 40, 55, 70, 120],
    labels=['18-40', '41-55', '56-70', '71+']
)

print(cohort['race'].value_counts())
print(f"Total MIMIC-IV breast cancer patients: {len(cohort)}")
```

Expected cohort size: **5,000–10,000 patients** depending on MIMIC-IV version and filtering.

### 2.4 Feature Alignment

Wisconsin Diagnostic features (FNA cell nuclei measurements) do not exist in MIMIC-IV. Instead, use clinically available MIMIC-IV lab values and vitals as a **proxy feature set** for malignancy risk prediction:

| Proxy Feature | MIMIC-IV Source |
|---|---|
| Hemoglobin | labevents (itemid: 51222) |
| WBC count | labevents (itemid: 51301) |
| Albumin | labevents (itemid: 50862) |
| LDH | labevents (itemid: 50954) |
| ALP | labevents (itemid: 50863) |
| Age | patients.anchor_age |
| Gender | patients.gender |

**Important framing for the paper:** The feature mismatch between Wisconsin Diagnostic and MIMIC-IV is the point — it creates a distributional shift that tests how DCGS behaves when models are applied outside their training distribution. Frame this explicitly as a stress-test design, not a limitation to hide.

Apply the Wisconsin-trained `StandardScaler` to MIMIC-IV features before running predictions.

### 2.5 Subgroup Definitions for DCGS

| Subgroup Axis | Categories | Minimum n Required |
|---|---|---|
| Race | White, Black, Hispanic, Asian, Other | ≥ 100 per group |
| Age tier | 18–40, 41–55, 56–70, 71+ | ≥ 100 per group |
| Insurance (SES proxy) | Medicare, Medicaid, Private, Other | ≥ 100 per group |

Use single-axis subgroups (race only, age only, insurance only) rather than intersectional combinations — MIMIC-IV breast cancer sample size won't support stable ECE estimates for race × age × insurance cells. Be explicit about this in the limitations.

---

## Section 3: Model Training (Wisconsin Diagnostic)

### 3.1 Models to Train and Calibrate

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

models = {
    'LR':  LogisticRegression(C=1.0, random_state=42),
    'RF':  RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, kernel='rbf', random_state=42),
    'NB':  GaussianNB(),
    'XGB': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}
```

Use 5-fold stratified cross-validation on the training set for hyperparameter tuning (GridSearchCV, optimizing AUROC). Report best hyperparameters in supplementary materials.

### 3.2 Calibration Methods

Apply and compare on the Wisconsin test set:
1. **No calibration** (baseline)
2. **Global Platt scaling** — `CalibratedClassifierCV(method='sigmoid', cv='prefit')`
3. **Global isotonic regression** — `CalibratedClassifierCV(method='isotonic', cv='prefit')`

On MIMIC-IV, additionally apply:
4. **Subgroup-targeted Platt scaling** — separate sigmoid calibration per demographic subgroup

**Core finding to set up:** Global calibration reduces population ECE but may not reduce DCGS. Subgroup-targeted calibration is needed to reduce both.

---

## Section 4: Computing DCGS

### 4.1 ECE Per Subgroup

```python
import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)

# Compute DCGS on MIMIC-IV subgroups
ece_k = {}
for subgroup_name, subgroup_mask in subgroups.items():
    if subgroup_mask.sum() >= 100:
        ece_k[subgroup_name] = compute_ece(
            y_true[subgroup_mask], y_prob[subgroup_mask]
        )

dcgs_range  = max(ece_k.values()) - min(ece_k.values())
dcgs_spread = np.std(list(ece_k.values()))
```

### 4.2 Statistical Testing

- **Bootstrap CI on DCGS** (1000 bootstrap samples of the MIMIC-IV test cohort): if 95% CI excludes 0, the gap is statistically significant
- **Pairwise ECE tests**: Wilcoxon signed-rank test between each pair of subgroups, Bonferroni-corrected

### 4.3 DCGS Results Table

| Calibration Method | Wisconsin ECE | MIMIC-IV ECE | MIMIC-IV DCGS (range) | MIMIC-IV DCGS (spread) |
|---|---|---|---|---|
| No calibration | X | X | X | X |
| Global Platt | X | X | X | X |
| Global isotonic | X | X | X | X |
| Subgroup Platt | — | X | X | X |

---

## Section 5: Reliability Diagrams

Generate calibration curves for:
1. **Wisconsin test set** — all 5 models, pre- and post-calibration (population-level, mirrors Paper 1 Figure 1)
2. **MIMIC-IV cohort** — best-calibrated variant per model, population-level (mirrors Paper 1 Figure 2)
3. **MIMIC-IV by subgroup** — ≥ 3 racial subgroups on the same plot to visually demonstrate DCGS

The subgroup reliability diagram (item 3) is the key new figure in this paper — it makes the demographic calibration gap visible in a way no prior breast cancer AI paper has shown.

---

## Section 6: Conformal Prediction — Subgroup Coverage

Extend Paper 1's conformal prediction framework:
- Apply split conformal prediction on Wisconsin test set (90% marginal coverage target)
- Apply to MIMIC-IV cohort — report per-subgroup coverage
- **Demographic Coverage Gap (DCG)** = max(coverage_k) − min(coverage_k); target < 0.05
- Expected finding: coverage transfers less reliably for underrepresented subgroups — a natural extension of Paper 1's collapse result into demographic dimensions

---

## Section 7: Impact Quantification

- Approximately **42 million** mammograms performed annually in the US (CDC, 2022)
- If AI-assisted risk scoring is applied to 10% of screening decisions = **4.2 million decisions/year**
- DCGS > 0.05 across racial subgroups means predicted probabilities are systematically off by ≥ 5% for specific demographic groups — affecting screening and biopsy thresholds
- Frame explicitly: this disparity is not evenly distributed; Black and Hispanic women, who already face documented disparities in breast cancer mortality, are at highest risk from miscalibrated AI risk scores

---

## Section 8: Open-Source Artifact

### 8.1 Package Structure

```
dcgs/
├── dcgs/
│   ├── metric.py            # DCGS computation (range + spread versions)
│   ├── subgroup_ece.py      # Per-subgroup ECE with bootstrap CI
│   ├── calibration_plot.py  # Multi-subgroup reliability diagram generator
│   └── conformal.py         # Demographic coverage gap for conformal sets
├── examples/
│   ├── wisconsin_demo.ipynb  # End-to-end demo on Wisconsin Diagnostic
│   └── mimic_demo.ipynb      # External validation + DCGS on MIMIC-IV subgroups
├── tests/
└── README.md
```

### 8.2 CLI Usage

```bash
pip install dcgs

dcgs compute \
  --predictions predictions.csv \   # columns: y_true, y_prob, subgroup
  --subgroup-col race \
  --n-bins 10 \
  --output dcgs_report.json
```

---

## Section 9: Paper Structure

1. **Abstract** (250 words): Gap (3.2% calibration reporting), Wisconsin + MIMIC-IV design, DCGS metric, key findings, tool release
2. **Introduction**: Breast cancer AI stakes → calibration gap → demographic calibration disparity → DCGS
3. **Background**: Calibration in clinical AI; fairness metrics vs. calibration equity; Wisconsin Diagnostic and MIMIC-IV as complementary datasets
4. **Methods**: Wisconsin cohort (training), MIMIC-IV cohort (external validation + DCGS), feature alignment, model training, calibration methods, DCGS definition
5. **Results**:
   - Wisconsin internal calibration results (Table T1 — mirrors Paper 1)
   - MIMIC-IV external calibration results (Table T2 — ECE drift)
   - DCGS by subgroup and calibration method (Table T3 — key new table)
   - Subgroup reliability diagrams (Figure — key new figure)
   - Subgroup conformal coverage results (Table T4)
6. **Discussion**: DCGS threshold implications; global vs. subgroup calibration; clinical impact on screening equity
7. **DCGS Toolkit**: Open-source release
8. **Limitations**: Single-center MIMIC-IV; feature mismatch between datasets; subgroup sample sizes; no intersectional analysis
9. **Conclusion**: DCGS as a standardized metric; call for demographic calibration reporting in all clinical AI studies

**Target word count:** 5,000–7,000 words (*JAMIA Open*); up to 8,000 (*JBI*)

---

## Section 10: Relationship to Other Papers

| Paper | Connection to Paper 2 |
|---|---|
| Paper 1 (CKD) | Same calibration evaluation methodology; same UCI → MIMIC-IV external validation design; Paper 2 cites Paper 1's framework and extends it to demographic equity |
| Paper 6 (CRI) | CRI's subgroup stability dimension uses DCGS logic; Paper 2's toolkit feeds into Paper 6 |
| Paper 8 (DCD) | DCD is a more sophisticated multi-dataset calibration equity metric that directly extends DCGS — Paper 2 is DCD's foundation |
| Paper 10 (CADRI) | DCGS/DCD results feed into CADRI's Fairness subscore |

---

## Section 11: Checklist Before Submission

- [ ] Wisconsin Diagnostic downloaded and preprocessing pipeline complete
- [ ] MIMIC-IV full release access approved (PhysioNet)
- [ ] MIMIC-IV breast cancer cohort extracted (ICD-10 C50.x), demographics merged
- [ ] Subgroup sizes confirmed (≥ 100 per racial group, age tier, insurance group)
- [ ] Five models trained and cross-validated on Wisconsin training set
- [ ] Three calibration conditions evaluated on Wisconsin test set (none, Platt, isotonic)
- [ ] Subgroup-targeted Platt scaling evaluated on MIMIC-IV subgroups
- [ ] DCGS computed (range and spread) for all models × calibration conditions on MIMIC-IV
- [ ] Bootstrap 95% CI on DCGS (1000 samples)
- [ ] Pairwise subgroup ECE significance tests (Bonferroni-corrected)
- [ ] Wisconsin reliability diagrams generated (population-level, mirrors Paper 1)
- [ ] MIMIC-IV subgroup reliability diagrams generated (≥ 3 racial subgroups on one figure)
- [ ] Conformal prediction coverage by subgroup; DCG reported
- [ ] Impact quantification: screening volume + DCGS > 0.05 threshold framing
- [ ] `dcgs` Python package released to GitHub under Apache 2.0, pip-installable
- [ ] Limitations section addresses: single-center MIMIC-IV, feature mismatch, subgroup sample sizes
- [ ] 3-reviewer internal review before submission

---

## Section 12: Timeline

| Milestone | Target |
|---|---|
| Wisconsin Diagnostic downloaded and preprocessing complete | Week 1 |
| MIMIC-IV full release access approved | Week 1–2 |
| MIMIC-IV breast cancer cohort extracted and cleaned | Week 2–3 |
| Five models trained, cross-validated, hyperparameters selected | Week 3–4 |
| Internal calibration evaluation on Wisconsin complete | Week 4 |
| MIMIC-IV external validation + DCGS computation complete | Week 5–6 |
| Subgroup-targeted calibration results complete | Week 6 |
| Conformal prediction subgroup coverage complete | Week 7 |
| Reliability diagrams and all figures finalized | Week 7–8 |
| `dcgs` package v1.0 released to GitHub | Week 8 |
| Paper draft complete | Week 8–10 |
| Internal review + revision | Week 10–11 |
| Submission to *JAMIA Open* | End of Week 12 |

> No data access waiting time beyond MIMIC-IV approval (~1–2 weeks). This paper can start immediately.

---

## Folder Structure

```
second_paper/
├── GUIDE.md                       ← this file
├── code/
│   ├── data_prep/                 ← Wisconsin download + MIMIC-IV cohort extraction
│   ├── modeling/                  ← model training, calibration, evaluation
│   ├── dcgs/                      ← DCGS metric implementation
│   ├── conformal/                 ← subgroup conformal prediction coverage
│   └── figures/                   ← reliability diagram generation scripts
├── data/
│   ├── wisconsin/                 ← Wisconsin Diagnostic (public, safe to commit)
│   └── mimic/                     ← MIMIC-IV breast cancer extract (add to .gitignore)
├── figures/
│   ├── reliability_wisconsin/     ← internal calibration curves
│   ├── reliability_mimic/         ← external calibration curves
│   └── dcgs_heatmap.png           ← DCGS by model × calibration method × subgroup
├── tables/
│   ├── T1_wisconsin_calibration.csv
│   ├── T2_mimic_calibration.csv
│   ├── T3_dcgs_results.csv
│   └── T4_conformal_coverage.csv
├── manuscript/
│   └── paper2_draft.docx          ← JAMIA Open Word submission
└── references/
    └── breast_cancer_calibration.bib
```

> **Data privacy note:** MIMIC-IV data is credentialed — add `data/mimic/` to `.gitignore` immediately. Wisconsin Diagnostic is public domain and safe to commit.
