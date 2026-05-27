# Paper 10 — Step-by-Step Execution Guide
### The Clinical AI Deployment Readiness Index (CADRI): A Validated Scoring Protocol for Production-Ready Healthcare AI Systems

**Author:** Michael O. Eniolade — University of the Cumberlands  
**Target Year:** Year 4  
**Primary Venue:** npj Digital Medicine (IF 15.2)  
**Backup Venue:** JAMIA or Frontiers in Digital Health  
**New Artifact:** Clinical AI Deployment Readiness Index (CADRI) — an empirically validated, scored composite index with open-source scoring tool

---

## One-Line Contribution

The first empirically validated, scored deployment readiness index for clinical AI — with data-driven subscore weights derived from real system performance data and demonstrated predictive validity (CADRI score predicts whether a system will fail on external validation) — bridging the AI chasm between model development and safe clinical deployment.

---

## Table of Contents

1. [Conceptual Foundation](#1-conceptual-foundation)
2. [CADRI Definition and Structure](#2-cadri-definition-and-structure)
3. [Validation Dataset: Systems from Papers 3, 6, 7, and 9](#3-validation-dataset-systems-from-papers-3-6-7-and-9)
4. [Empirical Weight Derivation](#4-empirical-weight-derivation)
5. [CADRI Threshold Validation](#5-cadri-threshold-validation)
6. [Comparing CADRI to Existing Checklists](#6-comparing-cadri-to-existing-checklists)
7. [Operationally Ready vs. Technically Ready](#7-operationally-ready-vs-technically-ready)
8. [Open-Source CADRI Tool](#8-open-source-cadri-tool)
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

### The AI Chasm

A 2025 PMC implementation framework paper identified a persistent "AI chasm" between model accuracy during development and clinical efficacy in deployment. Multiple 2024–2025 systematic reviews confirm: models that pass internal validation regularly fail in external clinical environments. The field lacks a standardized, scored, empirically validated protocol for determining whether a clinical AI system is ready to deploy.

Existing checklists (CASoF, AI-HIF) are expert-opinion-based governance documents. They tell you what to check, not how to score what you find, and their weights are not validated against real deployment outcomes.

### What CADRI Does Differently

CADRI is not a checklist — it is a scored index with empirically derived weights. The weights are determined by regressing CADRI subscores against real external validation failure rates across the clinical AI systems built and evaluated in Papers 3, 6, 7, and 9. This means CADRI's weights reflect what actually predicts deployment failure, not what experts believe should matter.

CADRI also introduces a key distinction that existing frameworks conflate:

- **Technically ready:** the model has good reliability metrics (CRI ≥ 70, DCD < 0.15)
- **Operationally ready:** the system has good governance, explainability usability, and security posture

A system can be technically ready but not operationally ready — and vice versa. CADRI makes this distinction explicit through two orthogonal subscore clusters, enabling targeted remediation.

---

## 2. CADRI Definition and Structure

### CADRI Composite Score

```
CADRI = w1·S_reliability + w2·S_security + w3·S_fairness + w4·S_explainability + w5·S_governance

Where all subscores are on a 0–100 scale:

S_reliability    = CRI from Paper 6 (composite of ECE, drift sensitivity, subgroup stability, uncertainty quality)
S_security       = Security posture score from Paper 3 adversarial testing protocol
S_fairness       = 100 - DCD_normalized (from Paper 8; 100 = perfect calibration equity; 0 = severe failure)
S_explainability = SHAP faithfulness score (0–50) + Clinician usability rating (0–50)
S_governance     = Governance checklist score (0–100; see below)

Weights w1–w5: empirically derived in this paper via regression on external validation failure rates
Initial hypothesis: w1 (reliability) > w3 (fairness) ≈ w2 (security) > w4 (explainability) ≈ w5 (governance)

CADRI thresholds:
  CADRI ≥ 75:  conditional deployment recommendation (requires monitoring plan)
  CADRI 60–74: conditional — remediate identified weak subscores before deployment
  CADRI 40–59: not ready — significant remediation required
  CADRI < 40:  do not deploy
```

### S_governance Scoring (the Governance Checklist, 0–100)

This subscore operationalizes the "operationally ready" dimension. Scored as: (items passing / total items) × 100

Governance checklist items (20 items, 5 points each):

**Data governance (5 items):**
1. Data use agreement documented and current
2. De-identification protocol documented and verified
3. Data provenance trail from source to training set documented
4. Data versioning in place (training data can be exactly reproduced)
5. Patient consent/waiver documented where applicable

**Model governance (5 items):**
6. Model version control in place (git or MLflow)
7. Training configuration reproducible (fixed seeds, documented hyperparameters)
8. Model card completed (intended use, out-of-scope uses, known limitations)
9. Bias audit documented (subgroup performance analysis completed)
10. External validation completed on at least one dataset outside training distribution

**Deployment governance (5 items):**
11. Incident response plan documented (what happens when model fails)
12. Monitoring plan documented (who reviews alerts, at what cadence)
13. Rollback procedure documented and tested
14. User training materials completed for clinician end-users
15. Legal/IRB review completed for intended deployment context

**Post-deployment governance (5 items):**
16. Model performance review schedule defined (e.g., quarterly)
17. Data drift monitoring active with defined alert thresholds
18. Feedback loop defined (clinicians can flag unexpected predictions)
19. Retraining trigger criteria defined
20. Retirement criteria defined (when will this model be replaced)

---

## 3. Validation Dataset: Systems from Papers 3, 6, 7, and 9

### The CADRI Validation Design

CADRI requires real clinical AI systems to compute its scores against and validate its weights. The systems from this dissertation serve as the validation dataset:

| System | Papers | Tasks | External validation available? |
|---|---|---|---|
| CKD prediction baseline models (LR, RF, XGB, SVM, GNB) | Paper 1 | CKD diagnosis | Yes — MIMIC-IV external |
| CKD post-calibration models | Paper 1 | CKD diagnosis | Yes — MIMIC-IV external |
| Breast cancer models (Wisconsin + SEER) | Paper 2 | Breast cancer prediction | Yes — cross-dataset |
| ED triage models (debiased variants) | Paper 4 | Emergency triage | Yes — held-out MIMIC-IV-ED |
| ICU mortality models (cross-site) | Paper 5 | ICU mortality | Yes — eICU held-out sites |
| TrustMed integrated system | Paper 9 | CKD + Triage + ICU | Yes — held-out data from all three |

This gives approximately 15–25 distinct model configurations with external validation outcomes — enough for regression-based weight derivation.

### What "External Validation Failure" Means for Regression

Define the binary outcome variable for weight derivation:
```
FAIL = 1 if ΔAUROC(external - internal) > 0.10 OR ECE_external > 0.20 OR DCD_external > 0.30
FAIL = 0 otherwise
```

This composite failure definition captures the three main deployment failure modes: discrimination loss, calibration failure, and calibration equity failure.

Regress FAIL against CADRI subscores (logistic regression) to derive empirical weights.

---

## 4. Empirical Weight Derivation

### Step 4.1 — Compute All CADRI Subscores for Each System

For each model/system in the validation dataset, compute:
- S_reliability: use `clinicalml-reliability` (Paper 6) to compute CRI on internal test data
- S_security: apply the Paper 3 security scoring protocol (automated where possible; documented checklist where manual)
- S_fairness: use DCD module (Paper 8) to compute DCD on demographic subgroups
- S_explainability: SHAP faithfulness score (automated); clinician usability rating (requires at least 3 rater scores per system — if clinical collaborators are available; otherwise use proxy measure: SHAP consistency score)
- S_governance: apply the 20-item governance checklist (Section 2)

### Step 4.2 — Logistic Regression for Weight Derivation

```
Fit: logit(P(FAIL)) = b0 + b1·S_reliability + b2·S_security + b3·S_fairness + b4·S_explainability + b5·S_governance

Convert logistic regression coefficients to weights:
  w_i = |b_i| / Σ|b_j|    (normalized absolute coefficients)
```

Report: direction of each coefficient (higher S_reliability should predict lower FAIL), magnitude, 95% CI via bootstrap, p-value.

### Step 4.3 — Handle Small Sample Size

With approximately 15–25 systems, sample size is limited. Address this:
1. Use leave-one-out cross-validation (LOOCV) for model evaluation — appropriate for very small N
2. Report bootstrap CIs on all weight estimates (1,000 bootstrap samples)
3. Frame this as a pilot empirical derivation — the weights are data-driven but acknowledge the sample size limit in the Discussion
4. Sensitivity analysis: show CADRI ranking of systems is stable across the plausible weight range (Monte Carlo weight variation)

---

## 5. CADRI Threshold Validation

### Step 5.1 — Validate the CADRI ≥ 75 Threshold

Using LOOCV:
- For each held-out system, predict FAIL using the CADRI score (derived on remaining systems)
- Compute sensitivity and specificity of the CADRI ≥ 75 threshold for predicting non-failure
- Compute AUC of the CADRI score as a continuous predictor of FAIL

### Step 5.2 — ROC Curve Analysis

Plot ROC curve of CADRI as a predictor of FAIL. Report:
- AUC
- Optimal threshold by Youden's J statistic
- Sensitivity and specificity at CADRI = 75

If the empirical optimal threshold differs from 75, report both and explain the discrepancy. Adjust the CADRI threshold recommendation if the empirical evidence suggests a different value.

### Step 5.3 — Calibration of CADRI as a Predictor

Calibration curve of CADRI score vs. observed FAIL rate. Show that CADRI is well-calibrated as a deployment readiness predictor (not just a classifier — the probability meaning matters).

---

## 6. Comparing CADRI to Existing Checklists

### Comparison Baselines

| Checklist | Description | Weakness |
|---|---|---|
| CASoF | Comprehensive AI Safety Framework | Expert-weighted, not empirically validated |
| AI-HIF | AI Health Impact Framework | Governance-focused; no reliability/calibration metrics |
| DECIDE-AI | Reporting guideline for AI-CDSS | Reporting standard, not deployment scoring |
| Random governance score | 20-item checklist equally weighted | No reliability or fairness metrics |

For each checklist, compute its score for each system in the validation dataset (to the extent the checklist items can be evaluated). Then compute each checklist's AUC as a predictor of FAIL.

**Expected finding:** CADRI has higher AUC than all governance-only checklists because it incorporates technical reliability and fairness metrics alongside governance items. This is the empirical case for CADRI's superiority.

---

## 7. Operationally Ready vs. Technically Ready

### Defining the Two Clusters

**Technical readiness cluster:** S_reliability + S_fairness (Papers 6 and 8)
**Operational readiness cluster:** S_security + S_explainability + S_governance (Papers 3 and 9)

Compute:
```
CADRI_technical    = (S_reliability + S_fairness) / 2
CADRI_operational  = (S_security + S_explainability + S_governance) / 3
```

### The Key Finding

Identify systems where CADRI_technical is high but CADRI_operational is low — "technically ready but not operationally ready." Show that these systems still fail external validation or have deployment-blocking governance gaps.

This finding proves the two dimensions are orthogonal: neither alone is sufficient. It also provides actionable guidance: a team with high CADRI_technical but low CADRI_operational knows to focus remediation on governance, security, and explainability — not model retraining.

### Clinical Meaning

A model with CADRI_technical = 85 and CADRI_operational = 40 should not deploy — not because the model is bad, but because the operational infrastructure is not ready. CADRI makes this visible and prevents the "good model, bad deployment" failure mode.

---

## 8. Open-Source CADRI Tool

### Tool Design

Release a Python command-line tool and web interface:

```
cadri-score \
  --model path/to/model.pkl \
  --test-data path/to/test.csv \
  --group-labels path/to/groups.csv \
  --governance-checklist path/to/governance.json \
  --security-report path/to/security_report.json \
  --output cadri_report.pdf
```

The tool automatically:
1. Loads the model and test data
2. Computes CRI (calls clinicalml-reliability)
3. Computes DCD (calls clinicalml-reliability fairness module)
4. Computes SHAP faithfulness score
5. Reads governance checklist responses from JSON
6. Reads security report scores from JSON
7. Applies empirical CADRI weights
8. Generates a PDF deployment readiness report with:
   - CADRI composite score
   - Subscore breakdown
   - CADRI_technical vs. CADRI_operational cluster scores
   - Deployment recommendation
   - Specific remediation recommendations for subscores below threshold

### Governance Checklist JSON Format

```json
{
  "data_governance": {
    "dua_documented": true,
    "deidentification_verified": true,
    "provenance_documented": false,
    "version_control_active": true,
    "consent_documented": true
  },
  "model_governance": { ... },
  "deployment_governance": { ... },
  "postdeployment_governance": { ... }
}
```

Any `false` item is automatically flagged in the CADRI report with a remediation recommendation.

---

## 9. Experiments

| Experiment | Description |
|---|---|
| E1 | Compute all CADRI subscores for all validation systems |
| E2 | Empirical weight derivation via logistic regression (LOOCV) |
| E3 | CADRI threshold validation (ROC, AUC, sensitivity/specificity at CADRI = 75) |
| E4 | CADRI calibration curve (predicted readiness vs. observed failure rate) |
| E5 | CADRI vs. existing checklists (AUC comparison for predicting FAIL) |
| E6 | Technical vs. operational readiness: identify mismatched systems |
| E7 | CADRI sensitivity to weight perturbation (Monte Carlo stability analysis) |
| E8 | CADRI tool demonstration: generate CADRI report for TrustMed system |

---

## 10. Tables and Figures Plan

### Tables

| # | Title |
|---|---|
| Table 1 | CADRI structure: five subscores, their sources, and scoring methods |
| Table 2 | Governance checklist: 20 items with pass/fail criteria |
| Table 3 | Validation systems: CADRI subscores and external validation outcomes (FAIL/PASS) |
| Table 4 | Empirical weight derivation: logistic regression coefficients with 95% CIs |
| Table 5 | CADRI vs. existing checklists: AUC comparison for predicting deployment failure |
| Table 6 | Technical vs. operational readiness: cluster scores for all validation systems |
| Table 7 | CADRI sensitivity analysis: score stability under weight perturbation |

### Figures

| # | Title |
|---|---|
| Figure 1 | CADRI framework: five subscores → composite score → deployment recommendation |
| Figure 2 | ROC curve: CADRI as predictor of external validation failure |
| Figure 3 | CADRI calibration curve: predicted readiness vs. observed failure rate |
| Figure 4 | Technical vs. operational readiness scatter: all validation systems |
| Figure 5 | CADRI weight sensitivity: radar chart of scores under weight variations |
| Figure 6 | Example CADRI report output (from open-source tool) for TrustMed system |

---

## 11. Paper Structure and Writing Guide

### npj Digital Medicine Format (main text ≤ 3,500 words)

#### Abstract (≤ 150 words, unstructured)
State the AI chasm problem. Name CADRI. State the empirical weight derivation. State the key finding (CADRI predicts external validation failure with AUC = X; outperforms existing checklists). State open-source tool release.

#### Introduction (approximately 600 words)
- Open with the AI chasm: the gap between development performance and clinical deployment outcomes
- Establish the inadequacy of existing checklists (governance-only, expert-weighted, not validated)
- Introduce CADRI; state its key differentiator: empirically derived weights, technical + operational readiness distinction
- State contributions as numbered list

#### Methods
- CADRI definition and subscore components
- Governance checklist (20 items)
- Validation systems (Papers 1–9 systems + external validation outcomes)
- Empirical weight derivation (logistic regression, LOOCV)
- CADRI threshold validation
- Comparison to existing checklists
- Open-source tool description

#### Results
- Empirical weights (Table 4): which subscores most predict deployment failure
- CADRI threshold validation (Figure 2, ROC analysis)
- CADRI vs. existing checklists (Table 5)
- Technical vs. operational readiness (Figure 4)
- CADRI tool example output (Figure 6)

#### Discussion
1. What the empirical weights reveal: reliability and fairness are the strongest predictors of deployment failure — reinforcing that governance-only checklists miss the key signals
2. The technical/operational distinction: clinical teams need to know which type of readiness to fix
3. Limitations of the validation: small N (15–25 systems); all from one research program; weights may shift in larger validation. Propose multi-site CADRI validation as future work
4. How CADRI fits into clinical AI procurement: propose CADRI ≥ 75 as a minimum standard for deployment approval in procurement contracts

#### Conclusion
CADRI is the first empirically validated deployment readiness index for clinical AI, with data-driven weights and demonstrated predictive validity. Open-source tool available for immediate use.

---

## 12. Novelty and Reviewer Defense

**If reviewers say "CADRI is just a weighted average of existing metrics:"**
> CADRI's contribution is three-fold: (1) the empirical weight derivation from real deployment outcomes — making CADRI the first data-driven deployment index rather than an expert-opinion checklist; (2) the technical vs. operational readiness distinction that provides actionable remediation guidance; (3) demonstrated predictive validity against external validation failure — CADRI is not just descriptive, it predicts outcomes. No existing checklist has been validated against real deployment failure outcomes.

**If reviewers question the small validation N:**
> Acknowledge this directly — small N is a limitation. Address it via bootstrap CIs and LOOCV. Propose multi-site CADRI validation study as next step. Show sensitivity analysis demonstrates weight uncertainty does not substantially change CADRI rankings. The goal here is to establish the CADRI methodology and demonstrate its validity; scaling validation is future work.

---

## 13. Software Stack

```
Python 3.11+
pandas, numpy
scikit-learn         # logistic regression for weight derivation
scipy                # bootstrap, ROC
matplotlib, seaborn
fpdf or reportlab    # PDF report generation
click                # CLI interface
streamlit            # optional web interface
pytest
```

### Integration with Other Packages

CADRI tool calls:
```python
from clinicalml_reliability import compute_cri, compute_dcd  # Papers 6, 8
from trustmed.security import security_score                   # Paper 3
```

---

## 14. Folder Structure

```
tenth_paper/
├── data/
│   ├── validation_systems/   # Subscore data for all 15–25 validation systems
│   └── external_outcomes/    # FAIL/PASS labels for each system
├── code/
│   ├── cadri/                # CADRI package source
│   │   ├── __init__.py
│   │   ├── scorer.py         # CADRI composite score computation
│   │   ├── governance.py     # Governance checklist scorer
│   │   ├── weights.py        # Empirical weight derivation
│   │   └── report.py         # PDF report generator
│   ├── 01_collect_subscores.py
│   ├── 02_weight_derivation.py
│   ├── 03_threshold_validation.py
│   ├── 04_checklist_comparison.py
│   └── 05_figures_tables.py
├── figures/
├── tables/
├── manuscript/
│   ├── main.tex
│   └── cover_letter.docx
└── references/
```

---

## 15. Timeline (Estimated 20 weeks)

| Weeks | Tasks |
|---|---|
| 1–3 | Collect and organize CADRI subscores for all validation systems from Papers 1–9 |
| 4–5 | Define FAIL outcome variable; verify external validation data for each system |
| 6–7 | Empirical weight derivation (logistic regression + LOOCV) |
| 8–9 | CADRI threshold validation (ROC analysis, calibration curve) |
| 10–11 | CADRI vs. existing checklists comparison |
| 12–13 | Technical vs. operational readiness analysis |
| 14–15 | CADRI sensitivity analysis; generate all tables and figures |
| 16 | Build CADRI open-source tool; test; generate example report |
| 17–18 | Write paper (npj Digital Medicine format) |
| 19 | Internal review; revise; prepare GitHub release |
| 20 | Final revision; release CADRI tool under Apache 2.0; submit |

---

## 16. Submission Checklist

- [ ] CADRI formally defined with equation in paper
- [ ] All five subscores computed for all validation systems
- [ ] Empirical weights derived with LOOCV and bootstrap CIs
- [ ] CADRI ROC analysis completed (AUC reported)
- [ ] CADRI calibration curve completed
- [ ] CADRI vs. existing checklists (CASoF, AI-HIF) AUC comparison completed
- [ ] Technical vs. operational readiness distinction demonstrated empirically
- [ ] CADRI sensitivity analysis under weight perturbation completed
- [ ] Open-source CADRI tool released on GitHub under Apache 2.0
- [ ] Example CADRI report included in paper (Figure 6)
- [ ] Governance checklist JSON schema documented
- [ ] Abstract ≤ 150 words
- [ ] Main text ≤ 3,500 words
- [ ] Ethics statement included
- [ ] Code repository linked in paper
- [ ] References formatted per npj Digital Medicine guidelines
