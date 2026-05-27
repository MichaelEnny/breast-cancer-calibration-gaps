# Paper 5 — Step-by-Step Execution Guide
### Diagnosing Cross-Site Degradation in ICU Risk Prediction: A Subgroup-Level Failure Attribution Analysis Using eICU

**Author:** Michael O. Eniolade — University of the Cumberlands  
**Target Year:** Year 2  
**Primary Venue:** npj Digital Medicine (IF 15.2)  
**Backup Venue:** Journal of Biomedical Informatics  
**New Artifacts:** Site-Subgroup Vulnerability Index (SSVI) + Subgroup Degradation Attribution Report (SDAR)

---

## One-Line Contribution

The first eICU study to introduce a validated cross-site vulnerability metric (SSVI) that quantifies which patient subgroups are most harmed by cross-site model deployment, and why — shifting the field from "models degrade" to "here is exactly who is harmed, by how much, and why."

---

## Table of Contents

1. [Conceptual Foundation](#1-conceptual-foundation)
2. [Dataset Access and Setup](#2-dataset-access-and-setup)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Defining the SSVI Metric](#4-defining-the-ssvi-metric)
5. [Patient Subgroup Construction](#5-patient-subgroup-construction)
6. [Cross-Site Transfer Methodology](#6-cross-site-transfer-methodology)
7. [SSVI Computation and Validation](#7-ssvi-computation-and-validation)
8. [Subgroup Degradation Attribution Report (SDAR)](#8-subgroup-degradation-attribution-report-sdar)
9. [Experiments](#9-experiments)
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

The literature already knows that ICU risk models degrade when transferred across hospital sites. AUROC drops of 0.10–0.20 are well documented. The gap this paper fills is: **which patients are harmed, by how much, and why?**

SSVI provides a single number per patient subgroup that answers: "If I deploy this model at a new site with known KL divergence in feature distributions, how much will this subgroup's prediction quality degrade per unit of distributional shift?"

A high-SSVI subgroup is a fragile subgroup — small distributional changes cause large performance drops. These are the patients a deployment team must monitor first.

### Clinical Stakes

- Approximately 5.7 million US ICU admissions annually
- ICU risk scores directly influence escalation decisions, resource allocation, and discharge timing
- Cross-site deployment without SSVI-informed site assessment means deploying to all patients equally when some subgroups face 2–3x higher degradation risk

### Key Literature Gaps Addressed

1. A 2024 Critical Care Medicine study (334,812 ICU stays) confirmed AUROC drops up to −0.200 but did not explain which subgroups drive the degradation
2. A 2024 study measured KL divergence shifts (meds 0.82, labs 0.58, vitals 0.34) between MIMIC and eICU but did not translate these to patient-level harm
3. No study provides a pre-deployment site assessment tool grounded in subgroup vulnerability scores

---

## 2. Dataset Access and Setup

### Primary Dataset: eICU Collaborative Research Database

- **Source:** PhysioNet (physionet.org/content/eicu-crd)
- **Access:** Credentialed PhysioNet account + CITI training + data use agreement (same as MIMIC-IV process)
- **Size:** 200,859 ICU admissions across 208 US hospitals, 2014–2015
- **Key advantage:** Multi-site — allows train-at-site-A / test-at-site-B analysis

### Secondary Dataset (Source Domain): MIMIC-IV

- Use MIMIC-IV as the source domain for model training
- Use eICU as the target domain (cross-site transfer)
- This replicates real deployment: train on a well-resourced academic center; deploy to diverse community hospitals

### eICU Key Tables

| Table | Contents | Role |
|---|---|---|
| `patient` | Demographics, admission/discharge, hospital ID, APACHE scores | Primary patient table |
| `vitalPeriodic` | Frequent vital sign measurements | Time-series features |
| `lab` | Laboratory results | Key feature set |
| `diagnosis` | ICD-9/10 codes | Subgroup definition (primary diagnosis) |
| `physicalExam` | Clinical exam findings | Supplementary features |
| `hospital` | Hospital characteristics (bed size, teaching status, region) | Site-level covariates |

### Target Variable

Primary outcome: **In-hospital mortality** (binary)
- Defined as: `patienthealthsystemstayid` linked to discharge status "Expired" in the `patient` table
- Secondary: ICU length of stay > 7 days (binary — as secondary validation outcome)

---

## 3. Data Preprocessing

### Step 3.1 — Site Selection

Select eICU hospitals for cross-site analysis:
- Minimum 300 admissions per hospital to ensure stable performance estimates
- This typically yields approximately 100–150 usable hospitals
- Group hospitals into: large academic (> 500 beds), medium community (200–500 beds), small community (< 200 beds)

### Step 3.2 — Feature Engineering

**Demographic features:** age (numeric and binned), gender, ethnicity (where available)

**Acuity features:** APACHE IVa predicted mortality, admission diagnosis group, admission source (ED, OR, floor, etc.)

**Vital signs (first 24h):** mean, min, max, SD for: heart rate, MAP, respiratory rate, temperature, SpO2, GCS score

**Labs (first 24h):** first and last values for: creatinine, BUN, sodium, potassium, bicarbonate, WBC, hemoglobin, platelet count, lactate, pH

**Comorbidity burden:** Elixhauser comorbidity index computed from ICD codes

### Step 3.3 — Missing Data Strategy

- Features with > 40% missingness across the dataset: drop and document
- Features with 10–40% missingness: retain with a missingness indicator variable added
- Features with < 10% missingness: median imputation within site (not global) — this preserves site-level variation that is part of the cross-site analysis

### Step 3.4 — Train/Test Protocol

- Source domain: MIMIC-IV ICU (MIMIC-IV `icu` module — `icustays`, `chartevents`, `labevents` tables)
- Target domain: eICU (multi-site)
- Strategy: Train exclusively on MIMIC-IV → evaluate on each eICU site separately → compute site-level and subgroup-level performance

---

## 4. Defining the SSVI Metric

### Formal Definition

For a patient subgroup G (e.g., elderly patients with sepsis) and a target site S:

```
SSVI(G, S) = ΔAUROC(G, S) / KL_divergence(G, S)

Where:
  ΔAUROC(G, S)      = AUROC_source(G) - AUROC_target_site_S(G)
                      (performance drop for subgroup G when deployed to site S)
  KL_divergence(G, S) = sum of KL divergences for key clinical features
                        for subgroup G between source and site S
                        (computed separately for vitals, labs, medications)

SSVI interpretation:
  High SSVI (> 0.15): subgroup is fragile — small feature shifts cause large AUROC drops
  Low SSVI (< 0.05):  subgroup is robust — performance holds across distributional shifts
```

### Why SSVI is Novel

Existing work reports either overall AUROC drop (not subgroup-specific) or overall KL divergence (not performance-linked). SSVI is the first metric to combine both into a per-subgroup vulnerability score that is actionable for deployment planning.

### Population-Averaged SSVI

Also compute `SSVI_avg(G)` = mean SSVI across all target sites for a given subgroup. This becomes the site-independent vulnerability score reported as the primary result.

---

## 5. Patient Subgroup Construction

### Subgroup Axes

Construct subgroups along four clinically meaningful axes:

**Age tier:** 18–44, 45–64, 65–79, 80+

**Acuity tier (APACHE IVa predicted mortality):** Low (< 10%), Medium (10–30%), High (> 30%)

**Primary diagnosis category (from ICD codes):**
- Sepsis/infection
- Cardiac (MI, CHF, arrhythmia)
- Respiratory (ARDS, pneumonia, COPD exacerbation)
- Neurological (stroke, seizure, TBI)
- Post-operative
- Other/mixed

**Comorbidity burden (Elixhauser index):** Low (0–2), Medium (3–5), High (> 5)

### Cross-Axis Subgroups

For SSVI computation, use single-axis subgroups first (more statistical power). Then compute SSVI for two-axis combinations where n ≥ 150 per subgroup. Document all cell sizes.

---

## 6. Cross-Site Transfer Methodology

### Step 6.1 — Source Model Training

Train three models on MIMIC-IV:
- Logistic Regression (interpretable baseline)
- XGBoost (primary model — best overall performance expected)
- MIMIC-IV APACHE IVa score as clinical baseline (not ML — already validated)

Hyperparameter tuning via 5-fold stratified cross-validation on MIMIC-IV training data. Lock final models before any eICU evaluation.

### Step 6.2 — Feature Distribution Measurement

For each eICU site and for each subgroup:
- Compute per-feature KL divergence between MIMIC-IV and eICU distributions
- Summarize as three composite KL scores: KL_vitals, KL_labs, KL_demographics
- Use KL_labs as the primary SSVI denominator (labs showed the highest shift in prior literature: 0.58)

### Step 6.3 — Site-Level Evaluation

For each eICU site (hospitals with ≥ 300 admissions):
- Compute overall AUROC and subgroup-level AUROC for each subgroup
- Compute ΔAUROC = AUROC_MIMIC − AUROC_eICU_site
- Compute SSVI per subgroup per site

### Step 6.4 — Aggregation

- Aggregate SSVI across sites to compute site-averaged SSVI per subgroup
- Rank subgroups by SSVI_avg — this produces the main result table
- Identify the top 5 most vulnerable and top 5 most robust subgroups

---

## 7. SSVI Computation and Validation

### Validation Approach

To confirm SSVI is a valid predictor of cross-site degradation, demonstrate:

1. **Prospective validation:** Hold out 20% of eICU sites as a validation set not used in SSVI computation. Show that SSVI_avg computed on the remaining 80% of sites correctly ranks subgroup vulnerability in the held-out sites (Spearman rank correlation ≥ 0.70).

2. **Threshold validation:** Define SSVI_avg > 0.10 as "high vulnerability." Show that high-vulnerability subgroups have significantly higher ΔAUROC in held-out sites than low-vulnerability subgroups (Wilcoxon rank-sum test, p < 0.05).

3. **Sensitivity to KL denominator:** Recompute SSVI using KL_vitals, KL_labs, KL_demographics separately. Report SSVI stability across denominator choices.

---

## 8. Subgroup Degradation Attribution Report (SDAR)

The SDAR is the practical, open-source output of this paper — a structured report format that a deployment team can generate before any cross-site ICU rollout.

### SDAR Contents

For a given source model and candidate target site, the SDAR automatically generates:

1. **Feature distribution comparison table:** KL divergences for top 20 features between source and target site
2. **Subgroup vulnerability table:** SSVI for each subgroup at this specific target site
3. **Predicted AUROC table:** Source_AUROC − (SSVI × estimated_KL_at_target) per subgroup
4. **Risk tier classification:** High risk (SSVI > 0.10), Medium risk (0.05–0.10), Low risk (< 0.05)
5. **Deployment recommendation:** Conditional (all subgroups low risk), Conditional with monitoring (some medium risk), Not recommended without local retraining (any high risk)

### Implementation

- Build SDAR as a Python function: `generate_sdar(source_model, source_data, target_data, subgroup_definitions)`
- Output: formatted PDF report + CSV tables
- Release as part of the open-source SSVI toolkit on GitHub

---

## 9. Experiments

| Experiment | Description | Expected Finding |
|---|---|---|
| E1 | Source model performance on MIMIC-IV by subgroup | Establishes baseline; APACHE IVa as clinical baseline comparison |
| E2 | Cross-site AUROC distribution across all eICU sites (overall) | AUROC drops of 0.08–0.18 expected; confirms known phenomenon |
| E3 | KL divergence measurement per feature per site | Labs and medications show highest site-level variation |
| E4 | Subgroup AUROC across eICU sites — identify worst-performing subgroups | Elderly high-acuity patients and sepsis patients expected as most fragile |
| E5 | SSVI computation for all subgroups | High SSVI subgroups identified; SSVI_avg ranking produced |
| E6 | SSVI prospective validation on held-out sites | Spearman correlation ≥ 0.70 confirms SSVI predictive validity |
| E7 | SDAR generation for three example target sites | Demonstrates practical utility of the tool |
| E8 | SSVI vs. overall site AUROC drop (correlation) | SSVI predicts site-level harm better than global KL divergence alone |

---

## 10. Tables and Figures Plan

### Tables

| # | Title |
|---|---|
| Table 1 | Dataset characteristics: MIMIC-IV vs. eICU patient population comparison |
| Table 2 | Feature-level KL divergences between MIMIC-IV and eICU (top 20 features) |
| Table 3 | Source model performance by subgroup (MIMIC-IV test set) |
| Table 4 | Cross-site AUROC drops: mean ± SD across eICU sites, by subgroup |
| Table 5 | SSVI ranking: top 10 most vulnerable and 10 most robust subgroups |
| Table 6 | SSVI prospective validation results (Spearman correlation, held-out sites) |
| Table 7 | SDAR example output for three representative eICU sites |

### Figures

| # | Title |
|---|---|
| Figure 1 | Study design: MIMIC-IV training → cross-site eICU evaluation → SSVI computation |
| Figure 2 | Heatmap: subgroup AUROC across eICU sites (subgroups × sites) |
| Figure 3 | Scatter plot: KL divergence vs. ΔAUROC per subgroup, colored by SSVI tier |
| Figure 4 | SSVI ranking bar chart: all subgroups ordered by SSVI_avg |
| Figure 5 | Prospective validation: predicted AUROC degradation vs. actual (held-out sites) |
| Figure 6 | Example SDAR visualization for one eICU site |

---

## 11. Paper Structure and Writing Guide

### npj Digital Medicine Format (main text ≤ 3,000 words; figures and tables without word limit)

#### Abstract (≤ 150 words, unstructured for npj DM)
Introduce the cross-site degradation problem. Name SSVI. State the main result (top vulnerable subgroups identified; SSVI validates with Spearman ρ ≥ X on held-out sites). State the practical output (SDAR tool released). State the clinical implication.

#### Introduction (approximately 500 words)
- Open with the well-known cross-site degradation problem
- Cite the 334,812-stay Critical Care Medicine study
- State what is missing: who is harmed and why
- Introduce SSVI in the last paragraph; state it fills the gap
- Note that the SDAR tool translates SSVI into deployment decisions

#### Methods
- Dataset (MIMIC-IV source + eICU target)
- Feature engineering and preprocessing
- SSVI formal definition (include equation box)
- Patient subgroup definitions
- Cross-site evaluation protocol
- SSVI validation methodology
- SDAR structure and generation

#### Results
- Overall cross-site performance degradation (confirms known phenomenon)
- Feature-level KL divergences
- Subgroup SSVI rankings (the main result)
- Prospective validation
- SDAR example outputs

#### Discussion
1. Which subgroups are most fragile and clinical interpretation (why are elderly sepsis patients high-SSVI?)
2. How SSVI changes deployment practice vs. current "evaluate globally then deploy" approach
3. SDAR as a standardized pre-deployment tool
4. Limitations: MIMIC-IV to eICU has a specific year gap (2014–2015 eICU vs. 2008–2019 MIMIC-IV); temporal shift confounds site shift

#### Conclusion
SSVI and SDAR provide the first validated tools for identifying at-risk patient subgroups before cross-site deployment. Open-source toolkit available.

---

## 12. Novelty and Reviewer Defense

**If reviewers say "AUROC drops are already known:"**
> The degradation is known. The contribution is the first subgroup-level attribution tool (SSVI) that quantifies vulnerability per patient group per unit of distributional shift, and the SDAR which translates this into an automated pre-deployment report. The field knows models degrade; this paper tells deployment teams who is harmed and how to check before deploying.

**If reviewers question KL divergence as the denominator for SSVI:**
> Report SSVI under three denominator variants (KL_vitals, KL_labs, KL_composite). Show SSVI rankings are stable (Spearman ρ > 0.85 between variants). KL_labs is used as primary because prior literature shows it has highest variance across sites.

---

## 13. Software Stack

```
Python 3.11+
pandas, numpy
scikit-learn
xgboost
scipy              # KL divergence computation (scipy.special.rel_entr)
statsmodels
matplotlib, seaborn
reportlab or fpdf  # SDAR PDF generation
jupyter
```

---

## 14. Folder Structure

```
fifth_paper/
├── data/
│   ├── raw/           # eICU and MIMIC-IV tables (do not commit)
│   └── processed/     # Site-partitioned cleaned datasets
├── code/
│   ├── 01_data_extraction.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_kl_divergence.py
│   ├── 05_model_training.py
│   ├── 06_cross_site_evaluation.py
│   ├── 07_ssvi_computation.py
│   ├── 08_ssvi_validation.py
│   ├── 09_sdar_generator.py
│   └── 10_figures_tables.py
├── figures/
├── tables/
├── manuscript/
│   ├── main.tex (or .docx)
│   └── cover_letter.docx
└── references/
```

---

## 15. Timeline (Estimated 22 weeks)

| Weeks | Tasks |
|---|---|
| 1–2 | Obtain eICU access; download and inspect tables; document hospital distribution |
| 3–5 | Preprocessing: MIMIC-IV source prep + eICU multi-site prep; feature engineering |
| 6–7 | KL divergence computation for all features across all eICU sites |
| 8–9 | Patient subgroup construction; source model training on MIMIC-IV |
| 10–12 | Cross-site evaluation; ΔAUROC computation per subgroup per site |
| 13–14 | SSVI computation and ranking; prospective validation on held-out sites |
| 15–16 | SDAR generator implementation; generate example reports for three sites |
| 17–18 | Generate all tables and figures |
| 19–20 | Write introduction, methods, results |
| 21 | Write discussion and conclusion; internal review |
| 22 | Final revision; format for npj Digital Medicine; release open-source toolkit; submit |

---

## 16. Submission Checklist

- [ ] eICU data use agreement signed and documented
- [ ] MIMIC-IV (source domain) access confirmed and preprocessing complete
- [ ] SSVI formally defined with equation in paper
- [ ] SSVI prospective validation completed (held-out sites; Spearman ρ reported)
- [ ] SSVI sensitivity to KL denominator reported
- [ ] SDAR generator released on GitHub under Apache 2.0
- [ ] Population-scale harm quantification included (5.7M US ICU admissions)
- [ ] All figures publication-quality (300 DPI minimum)
- [ ] Abstract ≤ 150 words (npj Digital Medicine format)
- [ ] Main text ≤ 3,000 words
- [ ] Ethics statement included
- [ ] Code repository linked in paper
- [ ] References formatted per npj Digital Medicine guidelines
