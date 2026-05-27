# Paper 8 — Step-by-Step Execution Guide
### Calibration Equity in Healthcare AI: A Demographic Subgroup Analysis of Miscalibration Harm in Triage and Diagnosis

**Author:** Michael O. Eniolade — University of the Cumberlands  
**Target Year:** Year 3  
**Primary Venue:** ACM FAccT (Conference on Fairness, Accountability, and Transparency)  
**Backup Venue:** AIES (AAAI/ACM Conference on AI, Ethics, and Society)  
**New Artifact:** Demographic Calibration Disparity (DCD) metric

---

## One-Line Contribution

The first paper to formalize "calibration equity" as a distinct dimension of algorithmic fairness — introducing the Demographic Calibration Disparity (DCD) metric and demonstrating that 30%+ of models with acceptable aggregate calibration exhibit clinically actionable DCD > 0.15 for at least one demographic subgroup.

---

## Table of Contents

1. [Conceptual Foundation](#1-conceptual-foundation)
2. [Defining the DCD Metric](#2-defining-the-dcd-metric)
3. [Datasets](#3-datasets)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Training Protocol](#5-model-training-protocol)
6. [DCD Computation and Threshold Validation](#6-dcd-computation-and-threshold-validation)
7. [Debiasing and Recalibration Experiments](#7-debiasing-and-recalibration-experiments)
8. [Experiments](#8-experiments)
9. [Tables and Figures Plan](#9-tables-and-figures-plan)
10. [Paper Structure and Writing Guide](#10-paper-structure-and-writing-guide)
11. [ACM FAccT Positioning](#11-acm-facct-positioning)
12. [Novelty and Reviewer Defense](#12-novelty-and-reviewer-defense)
13. [Software Stack](#13-software-stack)
14. [Folder Structure](#14-folder-structure)
15. [Timeline](#15-timeline)
16. [Submission Checklist](#16-submission-checklist)

---

## 1. Conceptual Foundation

### The Core Reframing

The fairness community has extensively studied whether AI models perform equally well (e.g., equal AUROC, equal precision) across demographic groups. This is *classification fairness*. An almost entirely unstudied question is whether models are *equally calibrated* across demographic groups — i.e., whether predicted probabilities accurately reflect true outcome frequencies for each group.

This distinction matters clinically: a model can pass equalized odds and still assign systematically over-confident probabilities to one demographic group, causing that group to receive inappropriate care thresholds.

### The Gap

A 2025 npj Digital Medicine scoping review found sensitivity gaps of > 31% for minority patients in critical care AI, but did not measure whether these gaps are driven by classification performance or calibration performance. No standardized metric exists for *calibration equity* — the extent to which miscalibration is equitably distributed.

### What DCD Provides

DCD is to calibration what equalized odds is to classification: a normalized, comparable measure of disparity. By releasing DCD as part of the Paper 6 `clinicalml-reliability` package, every future healthcare AI study can report calibration equity alongside standard fairness metrics.

### Why ACM FAccT is the Right Venue

ACM FAccT focuses on accountability and transparency in algorithmic systems. Calibration equity directly affects how trustworthy probability outputs are for each patient group — a core fairness and transparency concern. The conceptual contribution (naming and formalizing calibration equity as a new fairness dimension) is exactly the type of contribution FAccT values.

---

## 2. Defining the DCD Metric

### Formal Definition

```
DCD = max_{g ∈ G} |ECE(g)| - min_{g ∈ G} |ECE(g)|
      ──────────────────────────────────────────────
                     ECE_overall

Where:
  G            = set of demographic subgroups under analysis
  ECE(g)       = Expected Calibration Error for subgroup g
                 (computed using equal-width probability bins)
  ECE_overall  = ECE computed on the full test set

DCD interpretation:
  DCD = 0:      perfect calibration equity (all groups equally calibrated)
  DCD < 0.15:   acceptable calibration equity
  DCD 0.15–0.30: moderate calibration equity failure — warrants monitoring
  DCD > 0.30:   severe calibration equity failure — deployment concern
```

### Why Normalize by ECE_overall?

Normalizing by overall ECE makes DCD scale-invariant. A DCD of 0.20 means the worst-calibrated group has ECE that is 20% of the overall ECE worse than the best-calibrated group, relative to overall model calibration. This makes DCD comparable across models with different overall calibration quality.

### Alternative Formulation (also report)

```
DCD_max = max_{g ∈ G} |ECE(g) - ECE_overall|
```
This measures the maximum deviation of any subgroup's calibration from the overall model calibration. Report both DCD and DCD_max; discuss which is more actionable.

### Threshold Justification (the DCD > 0.15 claim)

The DCD > 0.15 threshold must be empirically derived and justified in the paper:
- Analyze the relationship between DCD and downstream clinical decision impact (e.g., false risk tier assignment rate)
- Show that DCD > 0.15 corresponds to clinically meaningful miscalibration in at least one clinical task
- Use the triage dataset: at DCD > 0.15, how many patients per 1,000 are assigned to the wrong urgency tier due to subgroup-specific miscalibration?

This empirical threshold derivation is what elevates DCD beyond an arbitrary definition.

---

## 3. Datasets

### Primary Dataset: MIMIC-IV-ED (triage task)

- Already accessed from Paper 4
- Outcome: ESI triage acuity (binarized: high vs. low acuity)
- Demographic variables: race, gender, age (from `edstays` table)
- Large enough for reliable subgroup ECE estimation with probability binning

### Secondary Dataset: UCI CKD (diagnosis task)

- Already preprocessed from Paper 1
- Outcome: CKD vs. not CKD
- Demographic variables: limited — age and some lab-derived proxies
- Used to demonstrate DCD in a smaller, simpler context (shows DCD is applicable even at small scale)

### Tertiary Dataset: eICU (diagnosis/severity task)

- Already accessed from Paper 5
- Outcome: in-hospital mortality
- Demographic variables: age, gender, ethnicity (available in eICU `patient` table)
- Multi-site nature allows DCD to be computed both within and across sites

### Why Three Datasets

Demonstrating DCD across three clinical tasks (triage, diagnosis, severity prediction) is essential to establish it as a general metric rather than a task-specific observation. ACM FAccT reviewers will expect generalizability evidence.

---

## 4. Data Preprocessing

### Subgroup Definition

For each dataset, define demographic subgroups consistently:

**Race/ethnicity:** White, Black/African American, Hispanic/Latino, Asian, Other/Unknown
**Age tier:** 18–39, 40–59, 60–74, 75+
**Gender:** Male, Female (note binary encoding limitation)

Minimum cell size for ECE computation: n ≥ 100 per subgroup per dataset. Document subgroups excluded due to small n.

### ECE Computation Requirements

ECE is sensitive to:
- Number of probability bins (use 10 equal-width bins; sensitivity-test with 5 and 15 bins)
- Minimum samples per bin (require ≥ 20 per bin; merge if fewer)
- Calibration method used (compute ECE before and after Platt scaling, isotonic regression)

Document all ECE computation choices explicitly in the methods section — reviewers will scrutinize this.

---

## 5. Model Training Protocol

### Models to Train (per dataset)

For each dataset, train:
1. Logistic Regression (inherently somewhat calibrated)
2. Random Forest (typically poorly calibrated without post-processing)
3. XGBoost (typically poorly calibrated without post-processing)
4. XGBoost + Platt scaling
5. XGBoost + Isotonic regression

This gives a range of calibration quality levels, allowing you to show DCD varies with overall model calibration quality.

### The Key Question

Does a model with good aggregate ECE (e.g., ECE < 0.05 overall) still show DCD > 0.15?

If yes — this is the core finding: aggregate calibration quality does NOT guarantee calibration equity.

If the answer is yes for ≥ 30% of the models you train, this validates the central claim.

### Train/Test Protocol

80/20 stratified split, stratifying on outcome AND major demographic group simultaneously. Ensures each subgroup is represented in both train and test sets.

---

## 6. DCD Computation and Threshold Validation

### Step 6.1 — Compute ECE Per Subgroup

For each model × dataset × subgroup combination:
- Compute ECE using 10 equal-width bins on the test set
- Compute 95% bootstrap confidence interval for ECE (1,000 bootstrap samples)
- Flag subgroups where the CI is too wide (small n) — report as "insufficient sample size"

### Step 6.2 — Compute DCD

For each model × dataset:
- Compute DCD using all subgroups with sufficient n
- Compute DCD for each demographic axis (race-only, age-only, gender-only)
- Report which axis shows the highest DCD — this identifies the primary equity concern

### Step 6.3 — Threshold Derivation

Derive the DCD > 0.15 threshold empirically:
1. For each triage model, compute DCD
2. For each demographic subgroup in the triage dataset, compute: wrong_tier_rate(g) = proportion of patients in group g incorrectly classified into wrong urgency tier due to model
3. Compute correlation between DCD and max_subgroup_wrong_tier_rate
4. Identify the DCD value at which wrong_tier_rate reaches clinical significance (e.g., 5% excess wrong-tier rate)
5. Report this as the empirically derived DCD > 0.15 threshold (adjust if empirical data suggests a different threshold)

### Step 6.4 — Relationship Between Overall ECE and DCD

The key analysis: scatter plot of ECE_overall vs. DCD for all model × dataset combinations.
- If DCD is uncorrelated with ECE_overall: calibration equity and aggregate calibration are independent properties — DCD adds information beyond what ECE_overall tells us
- Expected finding: moderate correlation but many models with low ECE_overall (good overall calibration) still show high DCD (poor calibration equity)

---

## 7. Debiasing and Recalibration Experiments

### Experiment: Does Recalibration Fix DCD?

Apply standard recalibration methods to models with DCD > 0.15:
1. Platt scaling (global — fits one sigmoid to all predictions)
2. Isotonic regression (global)
3. Group-specific Platt scaling (fits separate sigmoid per demographic group)
4. Group-specific isotonic regression

Measure DCD before and after each recalibration method.

**Expected finding:** Global recalibration reduces ECE_overall but does NOT substantially reduce DCD (because it adjusts all predictions equally, not subgroup-specifically). Group-specific recalibration does reduce DCD — but requires knowing group membership at inference time (which raises its own clinical implications).

This experiment shows that DCD > 0.15 is not a trivial problem that standard recalibration fixes — it requires deliberate group-aware intervention.

---

## 8. Experiments

| Experiment | Description | Expected Finding |
|---|---|---|
| E1 | DCD computation across all models and datasets | DCD > 0.15 in ≥ 30% of models with ECE_overall < 0.10 |
| E2 | DCD vs. ECE_overall scatter (the key diagnostic) | Low ECE_overall does NOT guarantee low DCD |
| E3 | DCD per demographic axis (race vs. age vs. gender) | Race axis expected to show highest DCD |
| E4 | DCD threshold derivation (triage dataset) | Empirical derivation of DCD > 0.15 as clinical threshold |
| E5 | Global recalibration vs. DCD | Global recalibration reduces ECE_overall but not DCD |
| E6 | Group-specific recalibration vs. DCD | Group-specific recalibration reduces DCD substantially |
| E7 | DCD across three datasets (generalizability) | DCD > 0.15 found in triage, diagnosis, and severity tasks |
| E8 | DCD sensitivity to ECE bin count (5, 10, 15) | DCD stable across bin counts (validates metric robustness) |

---

## 9. Tables and Figures Plan

### Tables

| # | Title |
|---|---|
| Table 1 | Dataset demographics: subgroup sizes and outcome rates for MIMIC-IV-ED, UCI CKD, eICU |
| Table 2 | ECE per subgroup for each model × dataset (the main calibration equity table) |
| Table 3 | DCD and DCD_max for all models across three datasets |
| Table 4 | Proportion of models with ECE_overall < threshold but DCD > 0.15 |
| Table 5 | Effect of recalibration methods on ECE_overall and DCD |
| Table 6 | DCD threshold derivation: DCD vs. wrong-tier rate (triage task) |

### Figures

| # | Title |
|---|---|
| Figure 1 | Calibration equity framework: from ECE_overall to DCD to intervention |
| Figure 2 | Reliability diagrams: best vs. worst calibrated subgroup for one model (the visual impact figure) |
| Figure 3 | Scatter plot: ECE_overall vs. DCD for all model × dataset combinations |
| Figure 4 | DCD by demographic axis (grouped bar chart: race vs. age vs. gender) |
| Figure 5 | Effect of recalibration on DCD: global vs. group-specific methods |
| Figure 6 | DCD threshold derivation curve: DCD vs. wrong-tier rate |

---

## 10. Paper Structure and Writing Guide

### ACM FAccT Format (approximately 9–10 pages + references)

#### Abstract (≤ 250 words)
Introduce calibration equity as an unstudied dimension of algorithmic fairness. Name DCD. State the empirical finding (X% of models with acceptable aggregate calibration show DCD > 0.15). State the recalibration finding (global methods fail to fix DCD). State open-source release.

#### Section 1 — Introduction
- Open with a patient-facing example: a triage model with ECE = 0.04 overall — sounds good. But for Black patients aged 60–74, ECE = 0.31. That means for this subgroup, a predicted 20% risk is actually a 51% risk. This patient is being systematically underestimated.
- Establish the gap: classification fairness is well-studied; calibration equity is not
- Name DCD; state contributions as a numbered list
- Note integration with Paper 6 (clinicalml-reliability) and Paper 4 (BPAS)

#### Section 2 — Background
- Calibration in ML (general)
- Fairness metrics in healthcare AI
- Why calibration equity is different from classification equity
- Prior work on demographic calibration (brief — this gap is largely empty, which justifies the paper)

#### Section 3 — DCD Metric Definition
- ECE review (brief)
- DCD formal definition (Equation 1)
- DCD_max alternative (Equation 2)
- Threshold derivation methodology

#### Section 4 — Datasets and Methods
- Three datasets described
- Model training protocol
- ECE computation details (bins, CI, minimum samples)
- Recalibration methods tested

#### Section 5 — Results
- DCD across models and datasets (Tables 2, 3)
- The aggregate vs. subgroup calibration paradox (Figure 3, Table 4)
- Recalibration experiment (Table 5, Figure 5)
- Threshold derivation (Table 6, Figure 6)

#### Section 6 — Discussion
1. The calibration equity paradox: why ECE_overall is insufficient (clinical implications)
2. Why global recalibration fails for DCD — root cause is distributional, not model-level
3. What "calibration equity" means as a policy standard for clinical AI procurement
4. Limitations: binary gender encoding, MIMIC-IV single-site origin for triage, ECE sensitivity to bin count (addressed in E8)

#### Section 7 — Conclusion
DCD formalizes calibration equity as the fairness dimension that classification metrics miss. Open-source as part of `clinicalml-reliability`. Recommend DCD reporting as a standard component of clinical AI fairness audits.

---

## 11. ACM FAccT Positioning

ACM FAccT evaluates papers on:
1. **Technical rigor** — DCD must be formally defined and empirically validated
2. **Accountability** — DCD gives healthcare AI developers a specific number to report and be accountable for
3. **Transparency** — DCD reveals something ECE_overall conceals
4. **Societal impact** — tie DCD directly to patient harm: how many patients per year receive systematically miscalibrated risk estimates

### Framing Advice

Do NOT frame this as "here is another metric." Frame it as: "the current practice of reporting only aggregate calibration is institutionally hiding harm to specific patient groups. DCD makes that harm visible, measurable, and reportable."

Use the opening example (the patient-facing miscalibration scenario from the introduction) as a recurring touchstone throughout the paper.

---

## 12. Novelty and Reviewer Defense

**If reviewers say "ECE per subgroup is obvious — why do you need a new metric?"**
> Computing ECE per subgroup individually does not provide a normalized, comparable summary of calibration equity. DCD normalizes by ECE_overall, making it comparable across models with different overall calibration levels. Without normalization, a model with ECE_overall = 0.20 and subgroup ECE = 0.30 looks different from one with ECE_overall = 0.02 and subgroup ECE = 0.12 — but DCD reveals the second model has worse calibration equity (DCD = 0.20 vs. 0.50 respectively for these examples).

**If reviewers question the 0.15 threshold:**
> The threshold is empirically derived (Section 6.3 / Experiment E4) — it is not arbitrary. We identify the DCD level at which subgroup-specific miscalibration exceeds a clinically meaningful wrong-tier rate in triage. If reviewers believe a different threshold is appropriate for different tasks, DCD is designed to be computed and interpreted continuously — the threshold is a reporting convenience, not a design constraint.

---

## 13. Software Stack

```
Python 3.11+
pandas, numpy
scikit-learn         # ECE via CalibrationDisplay; Platt/isotonic recalibration
xgboost
scipy                # bootstrap CI
matplotlib, seaborn
jupyter
```

DCD is implemented as a module in the `clinicalml-reliability` package from Paper 6:
```python
from clinicalml_reliability.fairness import DCD, DCD_max
score = DCD(y_true, y_pred_proba, group_labels)
```

---

## 14. Folder Structure

```
eighth_paper/
├── data/
│   ├── raw/           # MIMIC-IV-ED (from Paper 4), eICU (from Paper 5) — do not commit
│   └── processed/     # Subgroup-stratified test sets
├── code/
│   ├── 01_data_preparation.py
│   ├── 02_model_training.py
│   ├── 03_dcd_computation.py
│   ├── 04_threshold_derivation.py
│   ├── 05_recalibration_experiments.py
│   └── 06_figures_tables.py
├── figures/
├── tables/
├── manuscript/
│   ├── main.tex (or .docx)
│   └── cover_letter.docx
└── references/
```

---

## 15. Timeline (Estimated 18 weeks)

| Weeks | Tasks |
|---|---|
| 1–2 | Data preparation: pull subgroup-stratified test sets from Papers 4 and 5 data; UCI CKD from Paper 1 |
| 3–4 | Model training across three datasets; compute ECE per subgroup for all models |
| 5–6 | DCD computation; DCD vs. ECE_overall scatter; identify the "paradox" models |
| 7–8 | Threshold derivation experiment (triage task); derive and validate DCD > 0.15 |
| 9–10 | Recalibration experiment: global vs. group-specific methods |
| 11–12 | Generalizability across all three datasets; DCD by demographic axis |
| 13–14 | Generate all tables and figures |
| 15–16 | Write paper (ACM format); ACM FAccT has strict formatting requirements |
| 17 | Internal review; revise; integrate DCD into clinicalml-reliability package |
| 18 | Final revision; submit to ACM FAccT |

---

## 16. Submission Checklist

- [ ] DCD formally defined with equation in paper
- [ ] DCD_max alternative also defined and reported
- [ ] Threshold DCD > 0.15 empirically derived (not assumed)
- [ ] Results reported across three clinical datasets
- [ ] Core finding demonstrated: ≥ 30% of models with low ECE_overall show DCD > 0.15
- [ ] Recalibration experiment completed (global vs. group-specific)
- [ ] DCD released as module in `clinicalml-reliability` (Paper 6 package)
- [ ] Opening clinical example included (patient-facing miscalibration scenario)
- [ ] Population-scale harm quantification included
- [ ] Abstract ≤ 250 words
- [ ] Ethics statement included
- [ ] Code repository linked in paper
- [ ] References formatted per ACM FAccT guidelines
