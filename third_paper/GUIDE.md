# Paper 3 — Step-by-Step Completion Guide

**Title:** Evaluating Security-by-Design in Clinical AI: Adversarial Testing, Threat Assessment, and Security Posture Scoring

**Type:** Security evaluation + adversarial validation paper
**Target Venue:** *JAMIA* (primary) or *npj Digital Medicine* (if adversarial testing results are strong)
**Target Submission:** End of Year 2
**Open-Source Requirement:** Security posture scoring tool + adversarial test suite, Apache 2.0, GitHub

---

## What This Paper Must Prove

This paper answers a question no published clinical AI paper has answered: *How do you systematically measure whether a clinical AI system is actually secure?*

The contribution is not a design paper ("here is how to build a secure system") — it is an **evaluation paper** ("here is how to score security posture, here are the results of adversarial testing on a real clinical AI system, and here is the tool that lets anyone repeat this assessment"). The distinction matters for reviewers.

The central artifact is the **Security Posture Score (SPS)**: a quantified, reproducible metric (0–100) that summarizes how well a clinical AI deployment withstands healthcare-specific adversarial threats. SPS becomes one of five subscores in Paper 10's CADRI framework.

---

## Gap Statement (Use This in Your Introduction)

- A 2025 PMC review documented **624 hospital cyber-attacks in 2023** — double 2022's 304 — averaging **$1.3M disruption cost each**
- A 2025 JMIR systematic review explicitly identified the absence of implementation-level security integration in clinical AI systems
- MITRE ATT&CK for Healthcare exists but no clinical AI paper has mapped its attack vectors to a tested, quantified scoring framework for AI inference pipelines specifically
- The result: clinical AI systems are deployed into adversarial environments with **no standardized security evaluation protocol**

Your paper fills that gap.

---

## The New Metric: Security Posture Score (SPS)

**Definition:** SPS is a composite score (0–100) computed across four security evaluation dimensions:

| Dimension | Weight | What It Measures |
|---|---|---|
| **Adversarial Input Robustness** | 35% | Performance drop under FGSM-style perturbations on tabular clinical features (lab values, vitals, demographics) |
| **Threat Coverage** | 25% | Fraction of MITRE ATT&CK for Healthcare attack vectors for which architectural mitigations exist and are validated |
| **Access Control Integrity** | 20% | RBAC audit completeness: fraction of inference decisions with full, non-repudiable audit trail |
| **Compliance Posture** | 20% | HIPAA/HL7 FHIR checklist adherence score across data handling layers |

**Formula:**
```
SPS = 0.35 × (1 − mean_Δperformance_under_attack) × 100
    + 0.25 × (mitigated_attack_vectors / total_relevant_vectors) × 100
    + 0.20 × (audit_complete_decisions / total_decisions) × 100
    + 0.20 × (fhir_hipaa_checks_passed / total_checks) × 100
```

**Threshold:** SPS ≥ 65 recommended for conditional deployment; SPS ≥ 80 for production deployment recommendation.

**Why this is novel:** No healthcare-specific AI security metric that jointly measures adversarial robustness, threat coverage, audit completeness, and compliance posture currently exists in the literature.

---

## Section 1: Environment and Dataset Setup

### 1.1 What Clinical System Are You Evaluating?

Paper 3 evaluates the same CKD prediction models from Paper 1 (UCI CKD, n=400) plus a breast cancer classifier (Wisconsin Diagnostic, n=569). These are deliberately small and well-understood — the contribution is the **evaluation protocol**, not the prediction task. Using known models lets reviewers focus on the security methodology without debating the clinical task.

If time permits, extend evaluation to a MIMIC-IV-derived cohort (sepsis or readmission prediction, n > 10,000) as a third validation context.

### 1.2 Build the Evaluated System

You need a running inference system to attack. Use the following minimal stack:

```
FastAPI inference API (Python)
  → scikit-learn model (Random Forest, loaded from .pkl)
  → PostgreSQL database (patient records)
  → RBAC middleware (JWT-based, role: clinician / admin / auditor)
  → Audit logging (append-only log: timestamp, user_id, patient_id, prediction, confidence)
```

Docker Compose setup (minimum):
```yaml
services:
  api:       FastAPI + scikit-learn model
  db:        PostgreSQL
  audit_db:  Separate append-only audit store (PostgreSQL with write-only credentials for API)
  redis:     Session/token store
```

This system does not need to be production-ready. It needs to be testable. The evaluation protocol is the contribution, not the system itself.

---

## Section 2: Adversarial Input Testing (Dimension 1 — 35% of SPS)

### 2.1 Why Tabular Adversarial Testing Is Different

FGSM was designed for images. For tabular clinical data, you cannot compute gradients over raw pixel values. The adaptation for clinical tabular inputs:

1. **Identify sensitive input features** — features where small perturbations are clinically plausible (e.g., serum creatinine ±0.3 mg/dL, hemoglobin ±1.5 g/dL, blood pressure ±10 mmHg)
2. **Gradient-based perturbation**: For tree-based models (Random Forest, XGBoost), use a surrogate gradient via a local linear approximation (LIME-extracted coefficients) to determine perturbation direction
3. **Constraint-based perturbation**: Apply perturbations only within clinically plausible bounds (±2SD of feature distribution in training set)
4. **Boundary attack**: Move test samples toward the decision boundary iteratively; measure how many steps (perturbation magnitude) are required to flip the prediction

### 2.2 Metrics to Report

For each attack type and each model:
- **Prediction flip rate**: fraction of test samples whose prediction changed under attack
- **Mean perturbation magnitude required**: average L∞ norm of perturbation needed to flip prediction
- **AUROC drop under attack**: AUROC with clean inputs vs. AUROC with adversarial inputs (Δ = adversarial robustness gap)
- **Clinical plausibility rate**: fraction of adversarial inputs that fall within clinically plausible value ranges

Report these for: (1) no-defense baseline, (2) input validation defense (clipping to plausible bounds), (3) adversarial training defense (retrained with augmented adversarial examples).

### 2.3 Red Team Assessment

Run a simulated red team against the API layer (not just model inputs):

- **Prompt injection equivalent**: Submit malformed JSON payloads to the API — test whether the API returns error messages that leak model architecture or internal state
- **Privilege escalation**: Attempt to access admin endpoints with clinician JWT token — verify RBAC enforcement
- **Audit trail bypass**: Attempt to submit predictions without triggering audit log entries — verify log completeness
- **Replay attack**: Resubmit a valid token after session expiry — verify token invalidation

Document each test as pass/fail with evidence. This becomes the Threat Coverage subscore.

---

## Section 3: Threat Modeling — MITRE ATT&CK for Healthcare (Dimension 2 — 25% of SPS)

### 3.1 Map Relevant Attack Vectors

Go to [attack.mitre.org/matrices/enterprise/ics/](https://attack.mitre.org) and filter for healthcare-relevant tactics. The key ones for a clinical AI inference system:

| MITRE Tactic | Relevant Technique | Relevance to Clinical AI |
|---|---|---|
| Initial Access | Phishing: Spearphishing Link | Attacker targets clinical staff to gain API credentials |
| Execution | User Execution: Malicious File | Malicious model weights injected via file upload endpoint |
| Credential Access | Brute Force: Password Spraying | API key enumeration against authentication endpoint |
| Collection | Automated Collection | Bulk inference requests to extract model behavior via query attacks |
| Exfiltration | Exfiltration Over Web Service | Patient prediction outputs leaked via API response logging |
| Impact | Data Manipulation | Input feature tampering to force systematically wrong predictions for specific patient groups |

Identify 10–15 relevant techniques. For each, document:
- Whether your system architecture has a mitigation
- Whether the mitigation has been tested
- Result of the test (or "untested" if not feasible)

This produces the Threat Coverage dimension of SPS.

### 3.2 Focus Attack: Model Inversion / Membership Inference

Include at least one model privacy attack as a distinct experiment:
- **Membership inference**: Can an attacker determine whether a specific patient's record was in the training set by querying the prediction API? Use the Shokri et al. (2017) shadow model approach adapted for tabular data.
- Report: membership inference accuracy (baseline = 50% random). Anything above 60% represents a real privacy risk.

---

## Section 4: Access Control and Audit Integrity (Dimension 3 — 20% of SPS)

### 4.1 RBAC Implementation to Test

Define three roles:
```
clinician  →  can submit prediction requests, view own prediction history
admin      →  can manage users, view all predictions, cannot modify audit logs
auditor    →  read-only access to audit logs, cannot submit predictions
```

Test coverage:
- For every API endpoint, verify the role check is enforced with a negative test (wrong role → 403 returned)
- Verify audit log entries are written for 100% of prediction API calls (not just successful ones — failed attempts must also be logged)
- Verify audit log entries cannot be deleted or modified via any API endpoint

Report: audit completeness rate (target: 100%), RBAC enforcement pass rate (target: 100% of endpoints).

### 4.2 Non-Repudiation Test

Extract 50 random audit log entries. For each:
- Verify the log contains: timestamp, user_id, patient_id (hashed), input feature hash, model version, prediction output, confidence score
- Verify the entry cannot be forged by a clinician-role user (no write access to audit DB)
- Cross-check 10 entries against API server logs to verify no prediction was made without a log entry

---

## Section 5: HIPAA / HL7 FHIR Compliance Checklist (Dimension 4 — 20% of SPS)

### 5.1 HIPAA Safeguards Checklist

Score each item as Pass / Fail / Partial:

| Safeguard | Check |
|---|---|
| PHI encryption at rest | AES-256 applied to patient database |
| PHI encryption in transit | TLS 1.3 enforced on all API endpoints |
| Access logging | All PHI access events recorded with user ID and timestamp |
| Minimum necessary use | API only exposes features required for prediction, not full patient record |
| Audit controls | Non-repudiable log of all disclosures |
| Session management | JWT expiry ≤ 8 hours; refresh token rotation enforced |

### 5.2 HL7 FHIR Alignment

For the data ingestion layer:
- Input features align with FHIR Observation and Condition resource profiles
- Prediction outputs formatted as FHIR ClinicalImpression resource structure
- Document compliance percentage (items met / total applicable items)

---

## Section 6: Computing the Security Posture Score

Once you have results from all four dimensions, compute SPS per the formula in Section 0.

### Example Target Values

| Dimension | Target | Threshold |
|---|---|---|
| Adversarial Input Robustness | AUROC drop < 0.05 under constrained attack | < 0.10 drop = pass |
| Threat Coverage | ≥ 70% of relevant attack vectors mitigated and tested | ≥ 60% = conditional pass |
| Access Control Integrity | 100% audit completeness, 100% RBAC enforcement | ≥ 95% = pass |
| Compliance Posture | ≥ 85% HIPAA/FHIR checklist adherence | ≥ 75% = pass |
| **Overall SPS** | **Target ≥ 75** | **≥ 65 = conditional deployment** |

### SPS Sensitivity Analysis

After computing your baseline SPS:
1. Drop one defense at a time (e.g., remove input validation) and recompute — show how much each layer contributes
2. Compare SPS before and after adversarial training — quantify the improvement
3. Compute SPS for the "no defense" baseline — this establishes the lower bound and motivates the framework

---

## Section 7: Quantified Impact — The $1.3M Calculation

### 7.1 Attack Vector Cost Attribution

At 624 breaches × $1.3M average = $811M annual disruption cost for US healthcare (2023).

For each attack vector your architecture mitigates (from MITRE threat model):
1. Identify the fraction of reported breaches attributable to that vector class (use HHS OCR breach portal data for categories)
2. Apply to your tested mitigation effectiveness rate
3. Project: "If this security evaluation protocol had been applied to all hospital AI deployments, X% of breaches in attack category Y could have been detected earlier or prevented, representing $Z in potential disruption avoidance"

This is a conservative, bracketed estimate — not a guarantee. State it clearly as a projection.

### 7.2 Model Inversion Privacy Risk

For membership inference results:
- If your model achieves 65% membership inference accuracy, estimate the number of patients at risk from a healthcare AI system with an average training set size (e.g., 50,000 records in a typical MIMIC-IV study) — X patients whose membership status is detectable
- Frame this in terms of re-identification risk under HIPAA's Safe Harbor standard

---

## Section 8: Open-Source Tool — Security Posture Scoring Package

### 8.1 What to Release

Python package: `clinicalml-security` (or `ckd-security-eval` if scoped to the paper's datasets)

```
clinicalml_security/
├── adversarial/
│   ├── tabular_fgsm.py          # LIME-guided gradient perturbation for tabular models
│   ├── boundary_attack.py       # Iterative boundary attack for tabular data
│   └── membership_inference.py  # Shadow model membership inference test
├── threat_model/
│   ├── mitre_mapper.py          # Maps system components to MITRE ATT&CK techniques
│   └── coverage_scorer.py       # Computes threat coverage subscore
├── audit/
│   ├── rbac_tester.py           # Automated RBAC enforcement test suite
│   └── completeness_checker.py  # Audit log completeness verification
├── compliance/
│   └── hipaa_fhir_checklist.py  # HIPAA/FHIR checklist runner
├── sps.py                       # Main SPS computation (all four dimensions)
└── report.py                    # Generates structured HTML/PDF security evaluation report
```

### 8.2 CLI Interface (Make It Usable Without Code)

```bash
# Run full security evaluation
clinicalml-security evaluate \
  --model model.pkl \
  --test-data test.csv \
  --api-url http://localhost:8000 \
  --rbac-config rbac.json \
  --output-dir ./security_report/

# Output: security_report/sps_summary.json + security_report/full_report.html
```

### 8.3 Integration with clinicalml-reliability (Paper 6)

The SPS becomes one input to Paper 9's Framework Trustworthiness Score (FTS) and Paper 10's CADRI. Ensure `clinicalml-security` outputs a structured JSON report with a standard schema that `clinicalml-reliability` and the CADRI CLI can ingest.

---

## Section 9: Paper Structure

### Recommended Section Outline

1. **Abstract** (250 words): Gap, method, SPS metric, key results, tool release
2. **Introduction**: Healthcare breach statistics → lack of security evaluation standards → research question
3. **Background**: MITRE ATT&CK for Healthcare; tabular adversarial ML; clinical AI security gaps
4. **The Security Posture Score (SPS)**: Definition, four dimensions, formula, threshold rationale
5. **Study Design**: System under evaluation; datasets; evaluation protocol overview
6. **Adversarial Input Testing**: Methods, results table (AUROC drops, flip rates, by defense condition)
7. **Threat Modeling Results**: MITRE ATT&CK mapping table; coverage score; red team results
8. **Access Control and Audit Integrity Results**: RBAC enforcement results; audit completeness
9. **Compliance Posture Results**: HIPAA/FHIR checklist results
10. **Security Posture Score Results**: Composite SPS; sensitivity analysis; before/after defense comparison
11. **Impact Quantification**: $1.3M breach cost framework; privacy risk estimate
12. **`clinicalml-security` Tool**: Architecture; usage example; integration with downstream papers
13. **Limitations**: Simulated red team (not real attacker); single organization context; dataset scope
14. **Conclusion**: SPS as a reusable metric; call for standardized security evaluation in clinical AI

### Target Word Count: 7,000–9,000 words (JAMIA); 5,000–7,000 words (npj Digital Medicine)

---

## Section 10: Threat Modeling Reference — MITRE ATT&CK Worksheet

Use this as a starting template. Fill in Test Result column during evaluation:

| ID | Technique | System Layer | Mitigation | Tested? | Result |
|---|---|---|---|---|---|
| T1566 | Phishing: Spearphishing Link | Authentication | MFA enforcement | Yes | Pass |
| T1078 | Valid Accounts: Default Accounts | API | No default credentials | Yes | Pass |
| T1110 | Brute Force: Password Spraying | API | Rate limiting (10 req/min) | Yes | Pass |
| T1005 | Data from Local System | Database | Encryption at rest (AES-256) | Yes | Pass |
| T1071 | Application Layer Protocol | API | TLS 1.3 enforced | Yes | Pass |
| T1565 | Data Manipulation | Model Input | Input validation + bounds checking | Yes | Partial |
| T1530 | Data from Cloud Storage | Data Layer | RBAC + access logging | Yes | Pass |
| T1041 | Exfiltration Over C2 Channel | API Output | Output rate limiting | No | N/A |
| Custom | Model Inversion / Membership Inference | Model | Differential privacy (optional) | Yes | Partial |
| Custom | Adversarial Input: Gradient Attack | Model Input | Adversarial training | Yes | Pass (post-defense) |

---

## Section 11: Differentiation from Paper 9 (TrustMed)

This is important for reviewers — Paper 3 and Paper 9 both involve security. They are not the same paper:

| | Paper 3 | Paper 9 |
|---|---|---|
| **Question** | How do you *measure* a clinical AI system's security posture? | How do you *build* a trustworthy clinical AI system integrating all components? |
| **Contribution** | SPS metric + adversarial evaluation protocol | TrustMed six-layer integrated framework + FTS metric |
| **Scope** | Security evaluation only | Full trustworthiness integration (reliability + security + fairness + explainability) |
| **Output** | Security Posture Score for any existing system | An integrated system that *achieves* high FTS because SPS, CRI, DCD, etc. are all embedded |
| **Timeline** | Year 2 | Year 4 |

In Paper 9, you cite Paper 3's SPS as the validated security subscore feeding into FTS. Paper 3 is the *measurement tool*; Paper 9 is the *integrated system*.

---

## Section 12: Checklist Before Submission

- [ ] SPS metric formally defined with formula, four dimensions, and threshold justification
- [ ] Adversarial input testing: results for at least 2 models (CKD RF + breast cancer RF) and 3 defense conditions
- [ ] Membership inference attack: accuracy reported with risk estimate
- [ ] MITRE ATT&CK mapping: ≥ 10 techniques mapped, ≥ 8 tested
- [ ] RBAC enforcement: 100% endpoint coverage tested
- [ ] Audit log completeness: cross-validated against server logs
- [ ] HIPAA/FHIR checklist: every item scored with evidence
- [ ] SPS computed for: (a) no-defense baseline, (b) partial defenses, (c) full defense
- [ ] Sensitivity analysis: one-dimension-at-a-time ablation
- [ ] Impact quantification: breach cost projection with explicit assumptions
- [ ] `clinicalml-security` package: released on GitHub under Apache 2.0, pip-installable, README with CLI example
- [ ] Integration schema: SPS JSON output compatible with CADRI CLI (Paper 10)
- [ ] Related work section explicitly references MITRE ATT&CK for Healthcare literature AND clinical AI security reviews
- [ ] Limitation section addresses: simulated vs. real attacker, single organizational context, dataset scale
- [ ] 3-reviewer internal review completed before submission

---

## Section 13: Timeline

| Milestone | Target |
|---|---|
| System implementation (FastAPI + Docker Compose + RBAC) | Month 1–2 |
| Adversarial testing protocol designed and piloted on CKD data | Month 2–3 |
| MITRE ATT&CK threat model completed | Month 3 |
| Full adversarial test suite run on both datasets | Month 3–4 |
| Red team assessment completed | Month 4 |
| Audit trail and RBAC tests completed | Month 4 |
| HIPAA/FHIR checklist completed | Month 4 |
| SPS computed, sensitivity analysis completed | Month 5 |
| `clinicalml-security` package v1.0 released to GitHub | Month 5 |
| Paper draft complete | Month 5–6 |
| Internal review + revision | Month 6 |
| Submission to JAMIA | End of Month 6 |

---

## Folder Structure (Create These Subdirectories)

```
third_paper/
├── GUIDE.md                    ← this file
├── code/
│   ├── system/                 ← FastAPI + Docker Compose inference system
│   ├── adversarial/            ← adversarial test scripts
│   ├── threat_model/           ← MITRE ATT&CK mapping and coverage scorer
│   ├── audit/                  ← RBAC and audit completeness tests
│   ├── compliance/             ← HIPAA/FHIR checklist runner
│   └── sps/                    ← SPS computation and sensitivity analysis
├── data/
│   ├── ckd/                    ← UCI CKD dataset (from Paper 1)
│   ├── breast_cancer/          ← Wisconsin Diagnostic dataset
│   └── audit_logs/             ← generated audit log samples for testing
├── figures/
│   ├── sps_sensitivity.png     ← SPS by dimension, before/after defense
│   ├── adversarial_results.png ← AUROC drop by attack type
│   └── mitre_coverage_map.png  ← MITRE ATT&CK coverage heatmap
├── tables/
│   ├── sps_results.csv
│   ├── adversarial_results.csv
│   └── mitre_mapping.csv
├── manuscript/
│   └── paper3_draft.docx       ← JAMIA Word submission
└── references/
    └── security_papers.bib
```
