# Paper 7 — Step-by-Step Execution Guide
### Multi-Axis Clinical Failure Detection: A Joint Framework for Missingness, Distribution Shift, and Noise in Healthcare ML

**Author:** Michael O. Eniolade — University of the Cumberlands  
**Target Year:** Year 3  
**Primary Venue:** AAAI (Association for the Advancement of Artificial Intelligence)  
**Backup Venue:** CHIL (Conference on Health, Inference, and Learning)  
**New Artifact:** Multi-Axis Clinical Failure Detector (MACFD) — a new algorithm with theoretical detection power bounds

---

## One-Line Contribution

The first joint multi-axis clinical failure detector with theoretical detection power guarantees — a new algorithm (MACFD) that simultaneously tests for missingness-induced bias, covariate distribution shift, and input noise sensitivity, replacing sequential inconsistent single-axis monitoring in healthcare ML deployments.

---

## Critical Warning: AAAI Standard

AAAI is a top-3 AI conference with approximately 20% acceptance rate for technical papers. A paper that only applies existing tests (KS test, MMD, PSI, drift detectors) to clinical data will be rejected. This paper MUST introduce a new algorithm with:
1. A formal algorithmic description (pseudocode)
2. Theoretical detection power analysis (what shift magnitude does MACFD detect at given power/alpha?)
3. Proof or strong empirical evidence of lower false negative rate vs. sequential individual detectors
4. Empirical evaluation on ≥ 3 datasets

If execution scope is insufficient for AAAI, submit to CHIL instead — which accepts strong empirical framework papers without requiring formal theoretical bounds.

---

## Table of Contents

1. [Conceptual Foundation](#1-conceptual-foundation)
2. [The MACFD Algorithm Design](#2-the-macfd-algorithm-design)
3. [Theoretical Analysis](#3-theoretical-analysis)
4. [Datasets](#4-datasets)
5. [Experimental Protocol](#5-experimental-protocol)
6. [Baseline Comparisons](#6-baseline-comparisons)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Tables and Figures Plan](#8-tables-and-figures-plan)
9. [Paper Structure and Writing Guide](#9-paper-structure-and-writing-guide)
10. [Novelty and Reviewer Defense](#10-novelty-and-reviewer-defense)
11. [Software Stack](#11-software-stack)
12. [Folder Structure](#12-folder-structure)
13. [Timeline](#13-timeline)
14. [Submission Checklist](#14-submission-checklist)

---

## 1. Conceptual Foundation

### The Core Problem

Healthcare ML models deployed in real clinical environments face three simultaneous failure modes:
1. **Missingness shift:** the pattern of missing values changes (e.g., a hospital stops routinely ordering a lab test)
2. **Covariate shift:** the distribution of feature values changes (e.g., a new patient population, seasonal effects, protocol changes)
3. **Input noise:** random or systematic measurement errors increase (e.g., new equipment calibration, documentation error rate changes)

Current monitoring approaches test these one at a time. A 2025 ScienceDirect systematic review (32 studies) confirmed that single-axis monitoring is the norm. The problem: in real deployments, all three happen simultaneously, and testing them sequentially inflates false negative rates — you may detect none because each individual test falls below its threshold.

### What MACFD Does Differently

MACFD runs a single joint hypothesis test:

```
H0: No failures present (no missingness shift, no covariate shift, no noise increase)
H1: At least one failure type is present
```

By pooling evidence across all three axes simultaneously, MACFD has higher statistical power at the same false alarm rate than sequential testing, especially when multiple mild shifts co-occur.

### Motivating Example

Suppose each of missingness shift, covariate shift, and noise individually has a detection power of 0.40 at α = 0.05 with a given shift magnitude. Sequential testing would detect failure only if at least one test reaches its individual threshold — approximate detection probability ≈ 1 − (1−0.40)³ = 0.78. But this also inflates false alarm rate to approximately 3× α unless corrected. MACFD's joint test achieves detection probability ≥ 0.80 at a controlled α = 0.05 — the combination of evidence gives it higher power without inflating false alarms.

---

## 2. The MACFD Algorithm Design

### High-Level Design

MACFD has three components plus a joint decision layer:

**Component 1 — Missingness Detector (MD):**
- Compute per-feature missingness rate in monitoring window
- Compare to reference missingness rate using Fisher's exact test (for binary miss/observe)
- Output: test statistic T_MD per feature; summarize as T_MD_max = max over features

**Component 2 — Covariate Shift Detector (CSD):**
- For continuous features: two-sample Kolmogorov-Smirnov test vs. reference distribution
- For categorical features: chi-squared test of distributional homogeneity
- Summarize using Maximum Mean Discrepancy (MMD) across all features jointly: T_CSD = MMD(reference, monitoring_window)

**Component 3 — Noise Detector (ND):**
- Estimate noise level using prediction inconsistency under small perturbations: add ε-level Gaussian noise to each feature; measure change in predicted probability
- High noise sensitivity = model predictions vary substantially under small perturbations that should be clinically insignificant
- Output: T_ND = mean absolute change in predicted probability under δ-perturbations

**Joint Decision Layer:**
```
Given (T_MD, T_CSD, T_ND):
1. Standardize each statistic: Z_i = (T_i - μ_ref_i) / σ_ref_i
   (μ_ref and σ_ref estimated from reference window via bootstrap)
2. Compute joint test statistic: T_joint = f(Z_MD, Z_CSD, Z_ND)
   Candidate combining functions:
   - Soft combination: T_joint = w1·Z_MD + w2·Z_CSD + w3·Z_ND (linear)
   - Hard combination: T_joint = max(Z_MD, Z_CSD, Z_ND) (Simes correction)
   - Fisher combination: T_joint = -2·Σ log(p_i) (chi-squared under H0)
3. Reject H0 if T_joint > threshold_α
```

**Recommendation:** Use Fisher's combining method as primary (well-grounded theoretically) and report linear weighting as a variant. Use Benjamini-Hochberg correction for the combined p-value to control the false discovery rate.

### Algorithm Pseudocode (include in paper)

```
Algorithm MACFD(X_ref, X_monitor, model, α):
  Input:
    X_ref:     reference window feature matrix
    X_monitor: monitoring window feature matrix  
    model:     trained clinical ML model
    α:         target false alarm rate

  Step 1 — Compute component test statistics:
    T_MD  ← MissingnessDetector(X_ref, X_monitor)
    T_CSD ← CovariateShiftDetector(X_ref, X_monitor)
    T_ND  ← NoiseDetector(X_monitor, model, δ=0.01)

  Step 2 — Convert to p-values via bootstrap reference distribution:
    p_MD, p_CSD, p_ND ← BootstrapPValues(T_MD, T_CSD, T_ND, X_ref)

  Step 3 — Joint test via Fisher's method:
    T_joint ← -2 · (log(p_MD) + log(p_CSD) + log(p_ND))
    T_joint ~ χ²(6) under H0

  Step 4 — Decision:
    if T_joint > χ²_critical(6, α):
      return FAILURE_DETECTED, (p_MD, p_CSD, p_ND)
    else:
      return NO_FAILURE, (p_MD, p_CSD, p_ND)
```

---

## 3. Theoretical Analysis

### What You Need to Prove (or Empirically Demonstrate)

**Claim 1 — Type I error control:**
Show that under H0 (no failure), P(MACFD rejects) ≤ α.
- Proof approach: Fisher's method produces a chi-squared statistic under independence; derive or cite the result; note that under mild feature correlation, use simulation to show empirical Type I error ≤ α + δ for small δ

**Claim 2 — Detection power advantage over sequential testing:**
Show that for joint shifts of magnitude θ = (θ_MD, θ_CSD, θ_ND) where each individual shift is sub-threshold:
- P(MACFD detects | joint shift θ) > P(any individual detector detects | shift θ)
- Prove analytically for the case where individual test statistics are independent
- Demonstrate empirically (via simulation on clinical data distributions) for the correlated case

**Claim 3 — Minimum detectable shift:**
For a given dataset size n and target power β = 0.80 at α = 0.05:
- Derive the minimum shift magnitude θ_min that MACFD can detect
- Express as a function of n and the number of features
- This gives practitioners guidance: "MACFD needs at least N samples in the monitoring window to detect a shift of size θ"

**Approach if full proofs are infeasible:**
Substitute formal proofs with a comprehensive simulation study:
- Generate 1,000 datasets from calibrated clinical distributions at known shift magnitudes
- Show empirical detection rates vs. analytical predictions
- AAAI will accept strong simulation-based evidence if formal proofs are not provided for the correlated case

---

## 4. Datasets

### Clinical Datasets for Evaluation

| Dataset | Task | Used for |
|---|---|---|
| UCI CKD (400 patients) | CKD binary classification | Pilot — small scale, verify implementation |
| MIMIC-IV (ICU cohort) | Mortality prediction | Primary real-data evaluation |
| eICU (multi-site) | Cross-site deployment | Cross-site shift simulation |

### Shift Injection Protocol

For each dataset, inject controlled shifts to create ground-truth failure scenarios:

**Missingness shifts:**
- Inject: increase missingness rate by 10%, 20%, 30% in monitoring window for randomly selected features
- Label each window as shifted (ground truth = FAILURE_PRESENT)

**Covariate shifts:**
- Inject: shift mean of selected features by 0.5σ, 1.0σ, 1.5σ in monitoring window
- Use Gaussian mixture perturbation for more realistic shifts

**Noise shifts:**
- Inject: add Gaussian noise ε ~ N(0, σ) for σ = 0.1σ_feature, 0.3σ_feature, 0.5σ_feature

**Joint shifts:**
- Inject all three simultaneously at low magnitudes (each at 0.3σ level individually, which each individual detector fails to catch)
- This is the key scenario where MACFD's joint detection advantage is demonstrated

### Monitoring Window Sizes

Evaluate MACFD performance under window sizes n = [50, 100, 200, 500] to characterize minimum detectable shift as a function of n.

---

## 5. Experimental Protocol

### Experiment Design

For each dataset and each shift type/magnitude:
1. Divide data into: reference window (60%) + monitoring windows (40%, rolling)
2. Apply shift injection to monitoring windows (50% shifted, 50% unshifted — realistic 50% prevalence)
3. Run MACFD and all baseline detectors
4. Record: detection decision (0/1), p-values for each component, compute metrics

### Key Experiments

| Experiment | Description |
|---|---|
| E1 | Type I error verification: 1,000 unshifted monitoring windows; confirm empirical false alarm ≤ α + 0.01 |
| E2 | Individual shift detection: test each shift type alone at three magnitudes |
| E3 | Joint shift detection (the key experiment): all three shifts at sub-threshold magnitudes simultaneously |
| E4 | Minimum detectable shift curve: vary n from 50 to 500, record detection power |
| E5 | Comparison across three clinical datasets |
| E6 | Run MACFD on MIMIC-IV → eICU cross-site transfer (real shifts, no injection) |
| E7 | Compute CRI (from Paper 6 framework) pre/post MACFD detection — show integration |

---

## 6. Baseline Comparisons

### Baselines to Compare Against

| Baseline | Description |
|---|---|
| KS-test only | Two-sample KS test on each feature independently; Bonferroni correction |
| MMD only | Maximum Mean Discrepancy test on full feature matrix |
| PSI (Population Stability Index) | Standard industry drift detector |
| ADWIN | Adaptive Windowing drift detector |
| Sequential MACFD components | Run MD + CSD + ND sequentially with Bonferroni correction (shows value of joint test) |
| MACFD (proposed) | Joint test as defined |

### Key Comparison Metric

**False Negative Rate at controlled False Positive Rate (α = 0.05)**

The key claim: MACFD has lower FNR than all baselines at α = 0.05 in the joint shift scenario. This must hold across all three datasets.

---

## 7. Evaluation Metrics

**Detection performance (per experiment):**
- True Positive Rate (Detection Rate / Power) = P(detected | shift present)
- False Positive Rate = P(detected | shift absent) — must be ≤ α
- F1 score (detection)
- Area under ROC curve for detection decision

**Computational performance:**
- Runtime per monitoring window (milliseconds) — must be suitable for deployment
- Memory footprint

**Sensitivity analysis:**
- Detection power vs. window size n
- Detection power vs. shift magnitude θ
- Robustness to feature correlation (compare independent vs. correlated feature distributions)

---

## 8. Tables and Figures Plan

### Tables

| # | Title |
|---|---|
| Table 1 | MACFD component statistics and their null distributions |
| Table 2 | Type I error verification: empirical false alarm rate at α = 0.05 across datasets and window sizes |
| Table 3 | Detection power comparison: MACFD vs. baselines under individual shifts |
| Table 4 | Detection power comparison: MACFD vs. baselines under joint sub-threshold shifts (the key table) |
| Table 5 | Minimum detectable shift vs. window size n for each dataset |
| Table 6 | Real-data results: MACFD applied to MIMIC-IV → eICU cross-site transfer |

### Figures

| # | Title |
|---|---|
| Figure 1 | MACFD architecture: three-component diagram + joint decision layer |
| Figure 2 | Power curves: detection rate vs. shift magnitude for MACFD vs. baselines |
| Figure 3 | Joint shift scenario: MACFD vs. sequential testing (the key figure) |
| Figure 4 | Minimum detectable shift curve: power vs. window size n |
| Figure 5 | Real-data MIMIC-IV → eICU: MACFD detection timeline |

---

## 9. Paper Structure and Writing Guide

### AAAI Format (8 pages + references; strict formatting)

#### Abstract (approximately 150 words)
State the problem (sequential single-axis monitoring fails for joint shifts). Name MACFD. State the theoretical result (detection power bounds). State the empirical result (X% lower FNR vs. best baseline in joint-shift scenario). State open-source release.

#### Section 1 — Introduction
- Open with the deployment monitoring gap
- Establish the three failure modes; cite the ScienceDirect 2025 systematic review
- State why sequential testing fails for joint shifts (False Negative Rate analysis)
- Introduce MACFD; state the contributions as a numbered list:
  1. New joint detection algorithm (MACFD) with formal pseudocode
  2. Theoretical detection power bounds under H0 and H1
  3. Empirical evaluation on three clinical datasets
  4. Open-source Python package

#### Section 2 — Related Work
- Distribution shift detection in ML (general)
- Healthcare-specific drift detection
- Missingness in clinical data
- Hypothesis testing combination methods (Fisher, Simes, BH)
- Gap: no prior work combines all three axes with theoretical guarantees for clinical ML

#### Section 3 — Problem Formulation
- Formal definitions of the three failure types
- Joint null hypothesis
- Failure severity framework: mild (each axis sub-threshold) vs. severe (any axis above threshold)

#### Section 4 — MACFD Algorithm
- Component detectors (MD, CSD, ND)
- Statistical basis for each component
- Fisher combining method and null distribution
- Full pseudocode
- Computational complexity analysis

#### Section 5 — Theoretical Analysis
- Type I error proof/bound
- Detection power analysis for independent components
- Simulation-based results for correlated features
- Minimum detectable shift theorem

#### Section 6 — Experiments
- Datasets and shift injection protocol
- Baseline comparisons
- Results (Tables 2–6, Figures 2–5)

#### Section 7 — Discussion
- Why joint testing outperforms sequential testing (connect to theory)
- Practical guidance: minimum window size recommendation
- Integration with Paper 6 (clinicalml-reliability / CRI)
- Limitations: Fisher's method assumes component test statistics are independent; show sensitivity analysis

#### Section 8 — Conclusion
MACFD is the first clinical ML failure detector with joint theoretical power guarantees. Open-source package available.

---

## 10. Novelty and Reviewer Defense

**If reviewers say "KS test + MMD is sufficient:"**
> The joint shift scenario (Table 4, Figure 3) is the core empirical evidence. At shift magnitudes where each individual test achieves only 30–40% detection, MACFD achieves 75%+ — demonstrating that the joint test is not just different but strictly better in the clinically relevant scenario of multiple co-occurring mild shifts.

**If reviewers challenge the theoretical claims:**
> Present the simulation validation (E1 for Type I error; E4 for power curves). If formal proofs are not complete, be explicit: state which results are proven and which are supported by simulation. AAAI accepts simulation-based evidence if clearly labeled.

**If reviewers say "clinical datasets are too small:"**
> The theoretical analysis characterizes minimum detectable shift as a function of n. Small datasets demonstrate detection in a challenging regime. The eICU cross-site experiment (200K+ admissions) is the large-scale clinical validation. The theory explains why small n limits detection — this is a finding, not a flaw.

---

## 11. Software Stack

```
Python 3.11+
numpy, scipy        # statistical tests, KL divergence, KS test
sklearn             # MMD approximation via kernel methods
statsmodels         # Fisher combining
pandas
matplotlib, seaborn
pytest              # unit tests for MACFD package
jupyter
```

### Package Structure

Release as `macfd` Python package:
```
macfd/
├── __init__.py
├── detectors/
│   ├── missingness.py    # MD component
│   ├── covariate.py      # CSD component
│   └── noise.py          # ND component
├── joint.py              # MACFD joint test
├── utils.py              # bootstrap, standardization
└── tests/
    └── test_macfd.py
```

---

## 12. Folder Structure

```
seventh_paper/
├── data/
│   ├── raw/
│   └── processed/
├── code/
│   ├── macfd/             # Package source (mirrors above)
│   ├── 01_synthetic_experiments.py
│   ├── 02_clinical_experiments.py
│   ├── 03_baseline_comparisons.py
│   ├── 04_theoretical_validation.py
│   └── 05_figures_tables.py
├── figures/
├── tables/
├── manuscript/
│   ├── main.tex
│   └── appendix.tex
└── references/
```

---

## 13. Timeline (Estimated 24 weeks)

| Weeks | Tasks |
|---|---|
| 1–3 | Literature review: existing drift detectors, combining test methods, clinical shift studies |
| 4–6 | Algorithm design: MACFD components + Fisher combining; implement and unit-test each component |
| 7–8 | Theoretical analysis: Type I error proof; power simulation for independent components |
| 9–10 | Synthetic experiments: shift injection protocol; validate Type I error empirically |
| 11–13 | Clinical dataset experiments: UCI CKD + MIMIC-IV; individual and joint shift scenarios |
| 14–15 | eICU cross-site real-data experiment |
| 16–17 | Baseline comparisons; sensitivity analyses; minimum detectable shift curves |
| 18–19 | Generate all tables and figures |
| 20–21 | Write paper (AAAI LaTeX format); strict 8-page limit — cut aggressively |
| 22–23 | Internal review; revise; prepare supplementary material |
| 24 | Release macfd package on GitHub (Apache 2.0); submit to AAAI |

---

## 14. Submission Checklist

- [ ] MACFD algorithm described with formal pseudocode in paper
- [ ] Type I error verification completed (empirical false alarm ≤ α + 0.01)
- [ ] Joint shift detection experiment completed (Table 4 — the key result)
- [ ] Detection power bounds derived and reported
- [ ] Minimum detectable shift curve produced
- [ ] Baseline comparisons against ≥ 4 methods
- [ ] Results reported on ≥ 3 datasets
- [ ] `macfd` Python package released on GitHub under Apache 2.0
- [ ] Package has unit tests and documentation
- [ ] Paper formatted in AAAI LaTeX style
- [ ] Paper ≤ 8 pages (strict) + references
- [ ] Integration with Paper 6 (clinicalml-reliability) demonstrated in E7
- [ ] Ethics statement included
- [ ] All experiments reproducible with fixed seeds documented
