# Paper 9 — Step-by-Step Execution Guide
### TrustMed: A Validated Open-Source Framework for Trustworthy Clinical Decision Support Integrating Prediction, Reliability, Fairness, Explainability, and Security

**Author:** Michael O. Eniolade — University of the Cumberlands  
**Target Year:** Year 4  
**Primary Venue:** npj Digital Medicine (IF 15.2)  
**Backup Venue:** JAMIA  
**New Artifact:** TrustMed — a named, versioned, Docker-composable open-source clinical AI framework + Framework Trustworthiness Score (FTS)

---

## One-Line Contribution

The first fully implemented, validated, named clinical AI trustworthiness framework to integrate all five dimensions — prediction, reliability, fairness, security, and explainability — in a single open-source system with a quantified deployment readiness score (FTS), validated across three clinical tasks.

---

## CRITICAL NOTE: This Paper Must Be a System, Not a Survey

Paper 9 is the dissertation's flagship contribution. It will be rejected from npj Digital Medicine if it reads as a synthesis or review of Papers 1–8. The scientific contribution must be:

1. The TrustMed system itself — a functioning, deployed, tested software framework
2. The FTS metric — a novel composite score that quantifies deployment readiness
3. Empirical evidence that the integrated system produces outcomes that no individual component produces alone (the integration is the contribution)

---

## Table of Contents

1. [TrustMed System Architecture](#1-trustmed-system-architecture)
2. [The Framework Trustworthiness Score (FTS)](#2-the-framework-trustworthiness-score-fts)
3. [Datasets](#3-datasets)
4. [Layer-by-Layer Implementation Guide](#4-layer-by-layer-implementation-guide)
5. [Integration Experiments](#5-integration-experiments)
6. [Ablation Study](#6-ablation-study)
7. [FTS Computation and Validation](#7-fts-computation-and-validation)
8. [Tables and Figures Plan](#8-tables-and-figures-plan)
9. [Paper Structure and Writing Guide](#9-paper-structure-and-writing-guide)
10. [Novelty and Reviewer Defense](#10-novelty-and-reviewer-defense)
11. [Software Stack and Deployment](#11-software-stack-and-deployment)
12. [Folder Structure](#12-folder-structure)
13. [Timeline](#13-timeline)
14. [Submission Checklist](#14-submission-checklist)

---

## 1. TrustMed System Architecture

TrustMed is a six-layer system. Each layer has a specific technical role and maps to prior papers in the roadmap.

### Layer 1 — Data Layer

**Purpose:** Ingest, validate, and audit clinical data before it reaches the model

**Components:**
- FHIR R4-compatible data ingestion schema (supports EHR export formats)
- Schema validation with automatic rejection of records failing range checks
- Missingness profiling: automatic flagging of features with > 30% missingness in the incoming batch
- Audit log: every data record tagged with ingest timestamp, source, and schema version
- Patient de-identification check: verify no direct identifiers (name, MRN, DOB) in feature set

**Implementation:**
- Python module: `trustmed/data/ingestor.py`
- Schema defined as JSON Schema (FHIR-aligned)
- Uses `pydantic` for validation; `hashlib` for record-level audit trail

### Layer 2 — Processing Layer

**Purpose:** Preprocessing, feature engineering, and leakage prevention

**Components:**
- Preprocessing pipeline built as scikit-learn Pipeline objects (ensures train/test consistency)
- Imputation strategy selection: median (numeric), mode (categorical), or MICE (iterative) — configurable per dataset
- Feature scaling: StandardScaler for logistic regression; no scaling for tree models
- Categorical encoding: OrdinalEncoder for tree models; OneHotEncoder for linear models
- Temporal aggregation: rolling-window statistics for time-series features (first 24h for ICU; first 60min for ED)
- Anti-leakage enforcement: all preprocessing objects fit on training data only; inference pipeline loads fitted objects from disk

**Implementation:**
- `trustmed/processing/pipeline.py`
- Preprocessing objects serialized with `joblib`; versioned with dataset hash

### Layer 3 — Intelligence Layer

**Purpose:** Predictive modeling with configurable model selection

**Components:**
- Model registry: pre-configured support for Logistic Regression, Random Forest, XGBoost, LightGBM, MLP
- Pluggable design: new models added by implementing the `TrustMedModel` abstract interface
- Optional LLM-assisted reasoning module: calls a local LLM API (Ollama or similar) to generate differential diagnosis text from top SHAP features — clearly labeled as assistive, not diagnostic
- Ensemble option: majority voting or stacking of multiple models
- Model versioning: each trained model tagged with dataset hash, training date, feature set hash

**Implementation:**
- `trustmed/intelligence/model_registry.py`
- `trustmed/intelligence/llm_assistant.py` (optional; off by default)

### Layer 4 — Reliability and Trust Layer

**Purpose:** Real-time reliability monitoring using CRI and MACFD

**Components (from Papers 6 and 7):**
- CRI monitor: computes Clinical Reliability Index on incoming batches; alerts if CRI drops below 70
- MACFD monitor: runs Multi-Axis Clinical Failure Detector on each new monitoring window; alerts if joint failure detected
- Calibration monitor: computes ECE and DCD on recent predictions; alerts if DCD > 0.15 for any subgroup
- Drift monitor: runs MACFD's CSD component as a lightweight continuous monitor
- SHAP explainability: computes SHAP values for each prediction; stores top-5 features per prediction in the audit log

**Implementation:**
- `trustmed/reliability/cri_monitor.py`
- `trustmed/reliability/macfd_monitor.py`
- `trustmed/reliability/calibration_monitor.py`
- `trustmed/explainability/shap_explainer.py`

### Layer 5 — Security Layer

**Purpose:** Access control, audit trails, adversarial defense (from Paper 3)

**Components:**
- RBAC: role definitions (clinician, analyst, admin) with enforced permission sets
- API authentication: JWT tokens; API key rotation policy
- Audit trail: every inference request logged with: user role, input feature hash, prediction output, SHAP top features, timestamp
- Adversarial input detection: flag inputs where MACFD's ND component detects anomalous noise sensitivity
- Encryption: AES-256 for data at rest; TLS 1.3 for data in transit
- HIPAA checklist enforcement: startup check verifies required security configurations are active

**Implementation:**
- `trustmed/security/rbac.py`
- `trustmed/security/audit_logger.py`
- `trustmed/security/adversarial_detector.py`

### Layer 6 — Application Layer

**Purpose:** Clinician-facing interface with uncertainty-aware risk communication

**Components:**
- FastAPI REST API: `POST /predict` returns prediction + confidence interval + top SHAP features + DCD flag
- Clinician dashboard (lightweight web UI): shows risk score, calibration warning if DCD > 0.15, top explanatory features, audit trail
- Uncertainty communication: confidence intervals from conformal prediction (Paper 1 framework) shown alongside point estimate
- Alert system: pushes notification if CRI < 70 or MACFD detects failure in recent window

**Implementation:**
- `trustmed/api/app.py` (FastAPI)
- `trustmed/ui/dashboard.py` (Streamlit or similar lightweight UI)

---

## 2. The Framework Trustworthiness Score (FTS)

### Formal Definition

```
FTS = w1·CRI + w2·(100 - DCD_score) + w3·Security_Score + w4·XAI_Score

Where:
  CRI           = Clinical Reliability Index from Paper 6 (0–100)
  DCD_score     = 100 × DCD / DCD_max_threshold (maps DCD to 0–100 scale,
                  capped at 100 for DCD ≥ DCD_max_threshold)
  Security_Score = security posture score from Paper 3 adversarial testing (0–100)
  XAI_Score     = explainability quality score: SHAP faithfulness (0–50) 
                  + clinical usability rating (0–50) = 0–100
  
Weights (initial equal weighting — to be validated in Paper 10):
  w1 = w2 = w3 = w4 = 0.25

Interpretation:
  FTS ≥ 80:  high trustworthiness — conditional deployment recommended
  FTS 60–79: moderate — proceed with enhanced monitoring
  FTS < 60:  insufficient — do not deploy without framework remediation
```

### FTS Validation in This Paper

Demonstrate that FTS correlates with:
1. External validation AUROC drop (models with low FTS should show larger AUROC drop on held-out data)
2. CADRI score (Paper 10 will use FTS as one input to CADRI; show they are aligned)

Compute FTS for all models from Papers 1–8 where sufficient data exists. Show FTS ranking is consistent with qualitative assessment of each model's trustworthiness.

---

## 3. Datasets

### Three Clinical Tasks Required

| Task | Dataset | Outcome | Paper cross-reference |
|---|---|---|---|
| Disease diagnosis | UCI CKD | CKD vs. not CKD | Paper 1 |
| Emergency triage | MIMIC-IV-ED | High vs. low acuity (ESI) | Paper 4 |
| ICU severity | eICU | In-hospital mortality | Paper 5 |

All preprocessing follows the protocols from Papers 1, 4, and 5. No new preprocessing work required. Re-use processed data files.

---

## 4. Layer-by-Layer Implementation Guide

### Step 4.1 — Build and Test Each Layer Independently

Before integrating, each layer must have:
- Unit tests passing (pytest)
- A standalone demo notebook showing the layer working on a single dataset
- Documented API (what goes in, what comes out)

This modular approach ensures the integration can be debugged when issues arise.

### Step 4.2 — Layer Integration Order

Build in this order:
1. Data layer + Processing layer (no ML needed — pure data engineering)
2. Intelligence layer (connects to existing trained models from earlier papers)
3. Reliability layer (connects CRI and MACFD from Papers 6 and 7)
4. Security layer (wraps the stack with auth and audit)
5. Application layer (FastAPI + dashboard on top)

### Step 4.3 — Docker Compose Setup

```yaml
# docker-compose.yml
services:
  trustmed-api:
    build: .
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://trustmed:password@db:5432/trustmed
      - SECRET_KEY=${SECRET_KEY}
    depends_on: [db, redis]
  
  db:
    image: postgres:15
    volumes: ["pgdata:/var/lib/postgresql/data"]
  
  redis:
    image: redis:7
    # used for monitoring window buffering
  
  dashboard:
    build: ./ui
    ports: ["8501:8501"]
    depends_on: [trustmed-api]
```

### Step 4.4 — End-to-End Integration Test

Before running experiments, run end-to-end integration test:
1. Send 50 CKD test patients through the full pipeline
2. Verify: prediction received; SHAP values computed; CRI computed on the batch; audit log written; RBAC enforced
3. Inject a covariate shift; verify MACFD fires alert
4. Check DCD on the batch; verify dashboard shows DCD flag if DCD > 0.15

---

## 5. Integration Experiments

### Experiment 5.1 — Baseline vs. TrustMed System Behavior

For each of the three clinical tasks:
- Baseline: plain XGBoost model, no monitoring, no explainability, no security controls
- TrustMed: full six-layer system with the same underlying model
- Measure: prediction performance (identical — TrustMed wraps, not modifies, the model), CRI, DCD, FTS, and security posture

**Expected finding:** The model's discriminative performance is unchanged (TrustMed wraps the model, not modifies it). What changes is the reliability layer's ability to detect when performance is degrading and the fairness layer's ability to catch calibration equity failures before they reach patients.

### Experiment 5.2 — TrustMed Under Simulated Deployment Stress

Simulate four deployment scenarios:
1. Clean deployment (reference data matches training distribution)
2. Covariate shift (new hospital with different patient demographics)
3. Missingness shift (one lab test stops being ordered)
4. Noise injection (random sensor noise added to vital signs)

For each scenario, measure:
- Whether MACFD and CRI correctly flag the degradation
- Whether DCD remains controlled
- Time-to-detection (how many incoming patients before the alert fires)

### Experiment 5.3 — Ablation Study (Section 6)

Remove each layer one at a time; measure FTS and detection performance.

### Experiment 5.4 — FTS Correlation with External Validation Performance

Compute FTS for models from Papers 1–8 (where possible). Correlate FTS with:
- ECE on external validation data
- AUROC on external validation data
- DCD on external validation data

**Expected finding:** FTS predicts external validation performance better than any single component score (CRI, DCD, or Security_Score individually).

---

## 6. Ablation Study

The ablation study is required to demonstrate that each layer contributes to FTS and system safety. Remove each layer and report:

| Configuration | CRI | DCD | Security_Score | FTS | Detection Time |
|---|---|---|---|---|---|
| Full TrustMed | — | — | — | — | — |
| No Reliability Layer | — | — | — | — | — |
| No Fairness Monitoring | — | — | — | — | — |
| No Security Layer | — | — | — | — | — |
| No Explainability | — | — | — | — | — |
| No Data Validation | — | — | — | — | — |
| Baseline (no layers) | — | — | — | — | — |

**Key result:** Each layer's removal degrades FTS meaningfully — proving the integration is not redundant.

---

## 7. FTS Computation and Validation

### Step 7.1 — Compute FTS for Each Task

For each of the three clinical tasks, compute FTS under:
- Clean deployment condition
- Each stress scenario (Section 5.2)

Show FTS responds to deployment stress: FTS drops when model is in a degraded state.

### Step 7.2 — FTS Predictive Validity

Compute FTS on training+validation data. Then evaluate on held-out external validation (eICU held-out sites from Paper 5).

Show: models with higher FTS during development show smaller AUROC drop on held-out data.
Report: Pearson or Spearman correlation between FTS and ΔAUROC(held-out).

### Step 7.3 — FTS Threshold Calibration

For the three tasks, identify the FTS threshold above which models show ΔAUROC < 0.05 on held-out data (the acceptable degradation threshold). If this threshold is approximately 70–80, this supports the FTS ≥ 80 deployment recommendation.

---

## 8. Tables and Figures Plan

### Tables

| # | Title |
|---|---|
| Table 1 | TrustMed layer descriptions: inputs, outputs, components, paper origin |
| Table 2 | FTS computation components and current weight justification |
| Table 3 | Baseline vs. TrustMed: prediction performance + CRI + DCD + FTS across three tasks |
| Table 4 | Deployment stress scenarios: MACFD detection rates and time-to-detection |
| Table 5 | Ablation study results: FTS and detection performance by layer removal |
| Table 6 | FTS predictive validity: correlation with held-out AUROC drop |

### Figures

| # | Title |
|---|---|
| Figure 1 | TrustMed architecture diagram: six layers with data flows |
| Figure 2 | FTS components: stacked bar chart showing each component's contribution per model |
| Figure 3 | Deployment stress monitoring: FTS and CRI timeline under four stress scenarios |
| Figure 4 | Ablation study: FTS drop by layer removal |
| Figure 5 | FTS vs. ΔAUROC (held-out): scatter plot showing predictive validity |
| Figure 6 | Clinician dashboard screenshot showing TrustMed output for a sample patient |

---

## 9. Paper Structure and Writing Guide

### npj Digital Medicine Format (main text ≤ 3,500 words)

#### Abstract (≤ 150 words, unstructured)
State the integration gap. Name TrustMed and FTS. Summarize the ablation finding (each layer contributes). Summarize FTS validity (correlation with held-out performance). State open-source release.

#### Introduction (approximately 600 words)
- Open with the clinical AI deployment problem: high internal performance, poor real-world behavior
- Cite Papers 1–8 findings as the evidence base (this is your own work — use it as motivation)
- State the integration gap: components exist but no integrated framework validates them together
- Introduce TrustMed and FTS; state contributions as numbered list

#### Methods
- TrustMed architecture (reference Figure 1; keep description concise)
- FTS definition (include equation box)
- Datasets (three tasks — already preprocessed)
- Integration experiments (5.1–5.4)
- Ablation study design
- FTS validation methodology

#### Results
- Baseline vs. TrustMed (Table 3)
- Deployment stress detection (Table 4)
- Ablation study (Table 5, Figure 4)
- FTS predictive validity (Table 6, Figure 5)

#### Discussion
1. What the ablation proves: each layer earns its presence in the framework
2. What FTS predictive validity means: a high FTS during development predicts stable deployment
3. Practical path to TrustMed adoption: modular — teams can adopt individual layers before full integration
4. Limitations: tested on three retrospective datasets; prospective clinical trial required for deployment claims; equal FTS weights need empirical derivation (Paper 10 does this)

#### Conclusion
TrustMed is the first validated open-source system demonstrating that trustworthy clinical AI requires integrating prediction, reliability, fairness, security, and explainability — and that FTS quantifies deployment readiness. System available at [GitHub URL].

---

## 10. Novelty and Reviewer Defense

**If reviewers say "this is just a software paper:"**
> The scientific contribution is the FTS metric and its demonstrated predictive validity (FTS predicts external AUROC drop), and the ablation study proving that each layer contributes non-redundantly. The software is the vehicle for delivering and testing these scientific claims.

**If reviewers say "FTS weights are arbitrary:"**
> The initial equal weights are validated against held-out performance and shown to predict ΔAUROC. Paper 10 derives empirical weights from a larger validation study — this paper establishes the concept and demonstrates it works; Paper 10 refines the weighting. This is a natural two-paper sequence that npj DM reviewers should recognize.

---

## 11. Software Stack and Deployment

```
Python 3.11+
fastapi, uvicorn    # API layer
streamlit           # lightweight dashboard
sqlalchemy          # audit log persistence
postgresql          # database
redis               # monitoring window buffering
pydantic            # data validation
scikit-learn, xgboost, lightgbm
shap
docker, docker-compose
pytest              # testing
```

### GitHub Repository Structure

```
trustmed/
├── trustmed/
│   ├── data/
│   ├── processing/
│   ├── intelligence/
│   ├── reliability/
│   ├── security/
│   ├── explainability/
│   └── api/
├── ui/              # dashboard
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md        # extensive documentation
└── docs/            # API docs, architecture diagrams
```

---

## 12. Folder Structure

```
ninth_paper/
├── data/
│   ├── processed/   # Preprocessed datasets from Papers 1, 4, 5
├── code/
│   └── trustmed/    # Full TrustMed system source (mirrors GitHub repo)
├── experiments/
│   ├── 01_layer_tests/
│   ├── 02_integration_tests/
│   ├── 03_stress_scenarios/
│   └── 04_ablation/
├── figures/
├── tables/
├── manuscript/
│   ├── main.tex
│   └── cover_letter.docx
└── references/
```

---

## 13. Timeline (Estimated 30 weeks)

| Weeks | Tasks |
|---|---|
| 1–3 | Architecture finalization; GitHub repo setup; Docker environment |
| 4–6 | Data layer + Processing layer implementation and unit tests |
| 7–9 | Intelligence layer; load trained models from Papers 1, 4, 5 |
| 10–13 | Reliability layer: integrate CRI (Paper 6) and MACFD (Paper 7); unit tests |
| 14–16 | Security layer: RBAC, audit logging, adversarial detection (Paper 3 patterns) |
| 17–18 | Application layer: FastAPI + Streamlit dashboard |
| 19–20 | End-to-end integration testing; Docker Compose deployment test |
| 21–23 | Run integration experiments (baseline vs. TrustMed; stress scenarios) |
| 24–25 | Run ablation study; compute FTS; FTS predictive validity analysis |
| 26–27 | Generate all tables and figures |
| 28–29 | Write paper (npj Digital Medicine format) |
| 30 | Final revision; prepare GitHub release (Apache 2.0); submit |

---

## 14. Submission Checklist

- [ ] All six TrustMed layers implemented and tested
- [ ] Docker Compose deployment works end-to-end
- [ ] FTS formally defined with equation in paper
- [ ] Ablation study completed across all six layers
- [ ] FTS predictive validity computed (correlation with held-out ΔAUROC)
- [ ] Results validated across three clinical tasks (CKD, triage, ICU)
- [ ] GitHub repository released under Apache 2.0 with full documentation
- [ ] README includes quickstart, architecture diagram, and API reference
- [ ] Abstract ≤ 150 words (npj DM format)
- [ ] Main text ≤ 3,500 words
- [ ] Ethics statement included
- [ ] All datasets used are de-identified public or credentialed PhysioNet data
- [ ] Code repository linked in paper
- [ ] References formatted per npj Digital Medicine guidelines
