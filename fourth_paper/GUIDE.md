# Paper 4 — Step-by-Step Execution Guide
### Intersectional Fairness and Reliability in Emergency Department AI Triage: Mapping Demographic Bias Pathways in MIMIC-IV-ED

**Author:** Michael O. Eniolade — University of the Cumberlands  
**Target Year:** Year 2  
**Primary Venue:** CHIL (Conference on Health, Inference, and Learning)  
**Backup Venue:** JAMIA  
**New Artifact:** Bias Pathway Attribution Score (BPAS)

---

## One-Line Contribution

The first triage AI study to introduce a validated bias attribution metric (BPAS) that quantifies what proportion of intersectional performance gaps originate in upstream data collection disparities versus model inductive bias — traced across 160,000+ MIMIC-IV-ED visits.

---

## Table of Contents

1. [Conceptual Foundation](#1-conceptual-foundation)
2. [Dataset Access and Setup](#2-dataset-access-and-setup)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Defining the BPAS Metric](#4-defining-the-bpas-metric)
5. [Intersectional Group Construction](#5-intersectional-group-construction)
6. [Bias Pathway Tracing Methodology](#6-bias-pathway-tracing-methodology)
7. [Model Training and Evaluation](#7-model-training-and-evaluation)
8. [Experiments](#8-experiments)
9. [Debiasing Analysis](#9-debiasing-analysis)
10. [Tables and Figures Plan](#10-tables-and-figures-plan)
11. [Paper Structure and Writing Guide](#11-paper-structure-and-writing-guide)
12. [Novelty and Reviewer Defense](#12-novelty-and-reviewer-defense)
13. [Software Stack](#13-software-stack)
14. [Folder Structure](#14-folder-structure)
15. [Timeline](#15-timeline)
16. [Submission Checklist](#16-submission-checklist)

---

## 1. Conceptual Foundation

### The Core Reframing

Existing triage fairness literature asks: "Does the model perform worse for demographic group X?" (marginal fairness). This paper asks: "Why does the model perform worse — and is the root cause in the data before the model even trains, or in how the model learns?"

This is the BPAS contribution: decomposing the gap into an upstream fraction (data collection disparities) and a downstream fraction (model behavior).

### Key Gap

- A 2024 medRxiv study (n=160,016 MIMIC-IV-ED) found that marginal de-biasing dominates the literature, while intersectional fairness (race × gender × age simultaneously) is almost entirely unstudied in triage AI.
- A 2024 PMC study confirmed that racial differences in lab ordering in MIMIC-IV create a measurable bias pathway — but no study traces this pathway all the way through to triage decision disparities.

### What BPAS Enables

If BPAS is high (> 0.60): the dominant source of bias is upstream (differential lab ordering, documentation gaps). The intervention should be in data collection, not model post-processing.

If BPAS is low (< 0.40): the model is introducing or amplifying bias beyond what the data predicts. The intervention should be in the model.

This directs resources to the right fix — a clinical and policy contribution, not just a statistical one.

---

## 2. Dataset Access and Setup

### Dataset: MIMIC-IV-ED (Emergency Department module)

- **Source:** PhysioNet (physionet.org/content/mimic-iv-ed)
- **Access:** Requires credentialed PhysioNet account + CITI training completion + data use agreement
- **Size:** Approximately 425,000 ED stays; study uses visits with complete triage data — targets approximately 160,000 usable visits after exclusions

### Access Steps

1. Complete CITI Data or Specimens Only Research training (if not already done from Paper 1 MIMIC-IV access)
2. Apply for credentialed access at physionet.org — use your University of the Cumberlands affiliation
3. Download MIMIC-IV-ED module tables: `edstays`, `triage`, `vitalsign`, `medrecon`, `pyxis`, `diagnosis`
4. Store all raw files in `data/raw/` — never commit to version control

### Key Tables to Use

| Table | Contents | Role in study |
|---|---|---|
| `edstays` | ED stay metadata, demographics (race, gender, age), admission times | Primary demographic source |
| `triage` | Chief complaint, triage vitals, acuity/ESI score | Prediction target and features |
| `vitalsign` | Serial vital signs during stay | Feature engineering |
| `medrecon` | Medication reconciliation | Comorbidity proxies |
| `diagnosis` | ICD codes assigned at discharge | Outcome validation |

### Inclusion/Exclusion Criteria

- Include: adult ED visits (age ≥ 18), visits with complete triage vitals and ESI score, visits with demographic data (race, gender available)
- Exclude: pediatric visits, transfers, visits with missing ESI score (triage target), visits missing all demographic information

---

## 3. Data Preprocessing

### Step 3.1 — Demographic Encoding

**Race categories to use (standardize MIMIC-IV free-text):**
- White / Caucasian
- Black / African American
- Hispanic / Latino
- Asian
- Other / Unknown (as a separate category — do not drop)

Map all free-text race entries to these five categories using a lookup dictionary. Document all mapping decisions.

**Age grouping:**
- 18–39 (Young Adult)
- 40–59 (Middle Age)
- 60–74 (Older Adult)
- 75+ (Elderly)

**Gender:** Binary (Male / Female) as recorded — note limitation in discussion.

### Step 3.2 — Triage Feature Engineering

Features to extract from triage and vitals:
- Chief complaint (encode top 20 categories; group remainder as "Other")
- Arrival mode (ambulance, walk-in, etc.)
- Triage vitals: heart rate, systolic BP, diastolic BP, respiratory rate, temperature, SpO2
- Pain score
- Time of arrival (hour of day, day of week)
- First documented lab ordering flag (binary: yes/no within first 60 minutes)

### Step 3.3 — Lab Ordering Feature (the Bias Pathway Variable)

This is the key upstream variable for BPAS computation:
- For each demographic group, compute: proportion of visits where any lab was ordered within 60 minutes of triage
- This becomes the "upstream disparity" measure: differential_lab_rate[group] = P(lab ordered | group) − P(lab ordered | overall)

### Step 3.4 — Missing Data Handling

- Vitals with > 30% missingness: document and include a missingness indicator flag
- Impute remaining vitals using median imputation within demographic subgroups (not overall median — this preserves group-level signal)
- All imputation fit on training data only — no leakage

### Step 3.5 — Train/Test Split

- 70/15/15 stratified split (train/validation/test), stratifying on both ESI tier and race category
- Use the same random seed throughout — document seed in paper
- Final test set locked until all model selection is complete

### Step 3.6 — Target Variable Definition

Primary: ESI triage score (1–5, where 1 = most urgent)
- Frame as binary: ESI 1–2 (high acuity) vs. ESI 3–5 (lower acuity)
- Secondary: actual admission to hospital (binary) — for downstream outcome validation

---

## 4. Defining the BPAS Metric

### Formal Definition

For a given intersectional subgroup G (e.g., Black × Female × 60–74):

```
BPAS(G) = Upstream_Gap(G) / Total_Gap(G)

Where:
  Total_Gap(G)    = AUROC(overall) - AUROC(G)
  Upstream_Gap(G) = portion of Total_Gap attributable to differential lab 
                    ordering and documentation rates for group G
                  = estimated via mediation analysis (see Step 6.3)

BPAS ranges from 0 to 1:
  BPAS > 0.60  →  dominant source is upstream data disparities
  BPAS 0.40–0.60  →  mixed upstream and model sources
  BPAS < 0.40  →  dominant source is model behavior
```

### Why This is Novel

No existing fairness metric decomposes performance gaps by causal pathway. BPAS is the first metric to operationalize this decomposition in a clinical triage setting, making it actionable for both data governance teams and model developers.

### Validation Requirement

To validate BPAS, you must show that:
1. Groups with high BPAS (upstream-dominant) show reduced gaps when lab ordering disparities are synthetically equalized — but not when post-processing debiasing is applied
2. Groups with low BPAS (model-dominant) show reduced gaps when post-processing is applied — but not when upstream data is equalized
This cross-validation of intervention effectiveness confirms BPAS is measuring what it claims to measure.

---

## 5. Intersectional Group Construction

### Groups to Analyze

Construct intersectional subgroups as: Race × Gender × Age Tier

With 5 race categories × 2 gender × 4 age tiers = 40 potential subgroups.

Apply a minimum cell size of n ≥ 200 per subgroup for statistical validity. Document which subgroups fall below this threshold and exclude from BPAS computation (report as limitation).

### Marginal vs. Intersectional Comparison

For each metric (AUROC, false negative rate, ESI undertriage rate):
- Compute the marginal value (e.g., AUROC for all Black patients)
- Compute the intersectional value (e.g., AUROC for Black × Female × 60–74 patients)
- Report the difference — this is the "intersectional gap" that marginal analysis misses

The key finding will be that several subgroups with acceptable marginal fairness have severe intersectional fairness failures — this is the paper's central empirical result.

---

## 6. Bias Pathway Tracing Methodology

### Step 6.1 — Measure Upstream Disparities

For each demographic group, measure:
1. Lab ordering rate within 60 min of triage (primary upstream variable)
2. Imaging ordering rate within 60 min of triage
3. Physician contact time (time from triage to first physician note)
4. Documentation completeness score (proportion of triage fields completed)

Compute disparity as deviation from the overall population mean. Report as a table.

### Step 6.2 — Map Upstream Disparities to Missing Features

Show how differential lab ordering translates into differential feature missingness:
- For groups with low lab ordering rates, which model features are most frequently missing?
- Compute correlation between upstream_lab_rate(G) and feature_missingness_rate(G) for each feature

This is the "pathway" part of the paper: upstream behavior → missing features → degraded predictions.

### Step 6.3 — Mediation Analysis for BPAS Computation

Use causal mediation analysis to estimate what proportion of the total performance gap is mediated by the upstream disparity variables:

1. Fit outcome model: P(correct triage | features, group)
2. Fit mediator model: P(lab ordered within 60min | group, triage_vitals)
3. Use the product-of-coefficients method or bootstrapped indirect effects to estimate the mediated (upstream) vs. direct (model) portions of the gap
4. BPAS(G) = mediated_effect / total_effect

Use the `statsmodels` mediation module or the `causalml` library in Python.

### Step 6.4 — Sensitivity Analysis

Run BPAS computation under three different mediator specifications:
- Mediator 1: lab ordering only
- Mediator 2: lab ordering + imaging ordering
- Mediator 3: lab ordering + imaging + physician contact time

Report BPAS stability across specifications — this demonstrates robustness of the metric.

---

## 7. Model Training and Evaluation

### Models to Train

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline; coefficient analysis for bias |
| Random Forest | Nonlinear baseline; feature importance analysis |
| XGBoost | Primary performance model |
| XGBoost + Equalized Odds post-processing | Marginal debiasing comparison |
| XGBoost + Calibrated Equal Opportunity | Alternative debiasing comparison |

### Evaluation Metrics

**Overall performance:** AUROC, AUPRC, F1 (macro), Brier score

**Fairness metrics per intersectional subgroup:**
- Subgroup AUROC
- False Negative Rate (undertriage rate — clinically critical)
- False Positive Rate (overtriage rate)
- Equalized Odds difference
- Average Odds difference

**BPAS per subgroup** (the key contribution metric)

### Cross-Validation

5-fold stratified cross-validation on training data, stratifying on ESI tier + race. Report mean ± std for all metrics. Final evaluation on held-out test set only.

---

## 8. Experiments

| Experiment | Description | Expected Finding |
|---|---|---|
| E1 | Marginal vs. intersectional AUROC gaps | Intersectional gaps are larger and differently distributed than marginal gaps |
| E2 | Upstream disparity measurement (lab ordering, imaging, physician time) | Significant racial and age-related disparities in lab ordering rates |
| E3 | Upstream → missing feature pathway mapping | Lab ordering disparity correlates strongly with feature missingness for high-BPAS groups |
| E4 | BPAS computation across all valid subgroups | Subgroups partition into upstream-dominant (BPAS > 0.60) vs. model-dominant (BPAS < 0.40) |
| E5 | Marginal debiasing intervention on high-BPAS groups | Post-processing debiasing does NOT close the gap for upstream-dominant groups |
| E6 | Synthetic upstream equalization on high-BPAS groups | Equalizing lab ordering rates synthetically DOES close the gap for upstream-dominant groups |
| E7 | BPAS sensitivity analysis across mediator specifications | BPAS is stable (< 0.05 variation) across mediator definitions |

---

## 9. Debiasing Analysis

The key experiment for validating BPAS is showing that the correct intervention depends on BPAS score:

**For high-BPAS groups (upstream-dominant):**
- Apply: Synthetic upstream equalization — impute lab-derived features using population-level averages for groups with historically low lab ordering rates
- Measure: Change in subgroup AUROC and undertriage rate
- Expected: Gap closes significantly

**For low-BPAS groups (model-dominant):**
- Apply: Equalized odds post-processing on model output
- Measure: Change in subgroup AUROC and undertriage rate
- Expected: Gap closes significantly

**Cross-check (the key validation):**
- Apply high-BPAS intervention to low-BPAS groups → gap should NOT close
- Apply low-BPAS intervention to high-BPAS groups → gap should NOT close

This is the empirical proof that BPAS correctly identifies the intervention target.

---

## 10. Tables and Figures Plan

### Tables (CHIL limit: approximately 8 tables)

| # | Title |
|---|---|
| Table 1 | Dataset characteristics by demographic subgroup (n, % missing, lab ordering rate) |
| Table 2 | Upstream disparity measures by race, gender, and age group |
| Table 3 | Feature missingness rates by demographic subgroup |
| Table 4 | Overall model performance comparison (AUROC, F1, Brier) |
| Table 5 | Intersectional subgroup AUROC and undertriage rates (full table) |
| Table 6 | BPAS values for all valid intersectional subgroups |
| Table 7 | Effect of upstream equalization vs. post-processing debiasing on high-BPAS groups |
| Table 8 | BPAS sensitivity analysis across mediator specifications |

### Figures

| # | Title |
|---|---|
| Figure 1 | Study design flowchart: dataset → upstream measurement → BPAS → intervention |
| Figure 2 | Heatmap: intersectional AUROC gaps (race × age × gender) |
| Figure 3 | Bias pathway diagram: lab ordering disparity → missingness → prediction gap |
| Figure 4 | Scatter plot: upstream_disparity vs. AUROC_gap (colored by BPAS tier) |
| Figure 5 | BPAS distribution across all valid subgroups |
| Figure 6 | Debiasing intervention effectiveness by BPAS tier (grouped bar chart) |

---

## 11. Paper Structure and Writing Guide

### CHIL Format (6–8 pages main text + references)

#### Abstract (≤ 250 words, structured)
- Background: intersectional fairness in triage AI is unstudied; upstream bias pathways unmeasured
- Methods: MIMIC-IV-ED (n ≈ 160,000); intersectional group analysis; BPAS mediation analysis
- Results: X of Y subgroups show BPAS > 0.60 (upstream-dominant); post-processing fails for these groups; upstream equalization succeeds
- Conclusion: BPAS correctly identifies intervention targets; released as open-source toolkit

#### Section 1 — Introduction
- Open with the clinical stakes: 145 million US ED visits per year; undertriage directly increases mortality risk
- Establish the marginal vs. intersectional gap: cite the medRxiv 2024 study
- State the BPAS contribution in one paragraph — name it early
- End with: "The remainder of this paper is structured as follows..."

#### Section 2 — Background and Related Work
- Fairness in healthcare AI (general)
- Triage AI and MIMIC-IV-ED work
- Intersectional fairness theory (Crenshaw 1989; Buolamwini & Gebru 2018)
- Causal/mediation approaches to fairness (cite key causal fairness papers)
- Gap: no prior work combines intersectional fairness + causal pathway attribution + clinical triage

#### Section 3 — Dataset and Preprocessing
- MIMIC-IV-ED description, inclusion/exclusion, demographic distribution
- Upstream disparity variables defined
- Note IRB/data use agreement status

#### Section 4 — Methodology
- Intersectional group construction
- Upstream disparity measurement
- Mediation analysis for BPAS
- Model training protocol
- BPAS formal definition (include equation)

#### Section 5 — Results
- Upstream disparities (Table 2, Figure 3)
- Intersectional performance gaps (Table 5, Figure 2)
- BPAS results (Table 6, Figure 5)
- Debiasing experiment results (Table 7, Figure 6)

#### Section 6 — Discussion
Four moves:
1. What the BPAS results mean — which groups are upstream-dominant and why this is actionable
2. Why marginal debiasing fails for high-BPAS groups
3. Clinical and policy implications: data collection reform, not just model adjustment
4. Limitations: binary gender coding, MIMIC-IV single-institution origin, BPAS assumes mediation model is correctly specified

#### Section 7 — Conclusion
- BPAS is the first triage AI metric that tells you WHERE to intervene, not just that bias exists
- Open-source toolkit released

---

## 12. Novelty and Reviewer Defense

**If reviewers ask "what is novel about BPAS?"**
> BPAS is the first metric in clinical AI to decompose an intersectional performance gap into upstream (data) and downstream (model) sources using causal mediation analysis. No prior fairness metric answers the question "where does this bias come from and where should we fix it?" BPAS answers it with a single normalized number per subgroup.

**If reviewers ask "why does upstream equalization matter — isn't post-processing simpler?"**
> Post-processing debiasing can only re-rank model outputs; it cannot recover information that was never collected. For a patient whose lab values were never ordered due to implicit bias, no model intervention can compensate for that missing clinical information. BPAS identifies these cases precisely.

**If reviewers question mediation analysis assumptions:**
> Acknowledge the sequential ignorability assumption required for causal mediation. Report sensitivity analysis using E-values (VanderWeele & Ding 2017) to show robustness to unmeasured confounders.

---

## 13. Software Stack

```
Python 3.11+
pandas, numpy
scikit-learn
xgboost
statsmodels        # mediation analysis
causalml           # alternative causal estimation
shap               # feature attribution
matplotlib, seaborn
scipy
jupyter
```

---

## 14. Folder Structure

```
fourth_paper/
├── data/
│   ├── raw/           # MIMIC-IV-ED tables (do not commit)
│   └── processed/     # Cleaned, encoded datasets
├── code/
│   ├── 01_data_extraction.py
│   ├── 02_preprocessing.py
│   ├── 03_upstream_disparities.py
│   ├── 04_intersectional_groups.py
│   ├── 05_bpas_mediation.py
│   ├── 06_model_training.py
│   ├── 07_debiasing_experiments.py
│   └── 08_figures_tables.py
├── figures/
├── tables/
├── manuscript/
│   ├── main.tex (or .docx)
│   └── cover_letter.docx
└── references/
```

---

## 15. Timeline (Estimated 20 weeks)

| Weeks | Tasks |
|---|---|
| 1–2 | Obtain MIMIC-IV-ED access; extract and inspect data; document demographic distributions |
| 3–4 | Preprocessing: demographic encoding, feature engineering, missingness analysis |
| 5–6 | Upstream disparity measurement; compute differential lab ordering and documentation rates |
| 7–8 | Intersectional group construction; compute marginal vs. intersectional performance gaps |
| 9–11 | BPAS mediation analysis; validate across mediator specifications; sensitivity analysis |
| 12–13 | Model training; debiasing experiments; validate BPAS intervention predictions |
| 14–15 | Generate all tables and figures |
| 16–17 | Write introduction, background, methodology |
| 18–19 | Write results and discussion; internal review |
| 20 | Final revision; format for CHIL; prepare open-source toolkit; submit |

---

## 16. Submission Checklist

- [ ] MIMIC-IV-ED data use agreement signed and documented
- [ ] BPAS metric formally defined with equation in paper
- [ ] Mediation analysis sensitivity analysis (E-values) completed
- [ ] Intersectional groups with n ≥ 200 only — exclusions documented
- [ ] Debiasing cross-validation experiment completed (key validation)
- [ ] BPAS toolkit released on GitHub under Apache 2.0
- [ ] All figures publication-quality (300 DPI minimum)
- [ ] Abstract ≤ 250 words
- [ ] Ethics statement included (credentialed de-identified data)
- [ ] Conflict of interest statement included
- [ ] Code repository linked in paper
- [ ] References formatted per CHIL guidelines
