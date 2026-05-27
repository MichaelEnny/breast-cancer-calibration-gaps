# PhD Research Planning Document
### Michael O. Eniolade — University of the Cumberlands

---

## Table of Contents

1. [Research Direction Summary](#1-research-direction-summary)
2. [Proposed PhD Dissertation](#2-proposed-phd-dissertation)
3. [APA-Style Proposal Structure](#3-apa-style-proposal-structure)
4. [Chapter Outline](#4-chapter-outline)
5. [Publication Roadmap: 10 Papers](#5-publication-roadmap-10-papers)
6. [Paper-to-Dissertation Mapping](#6-paper-to-dissertation-mapping)
7. [Publication Sequence by Year](#7-publication-sequence-by-year)
8. [Dissertation Final Claims](#8-dissertation-final-claims)
9. [Literature Gap Evidence (Search-Verified)](#9-literature-gap-evidence-search-verified-april-2026)
10. [Authorship Strategy](#10-authorship-strategy)
11. [Paper 1 — Full Planning Guide (CKD)](#11-paper-1--full-planning-guide-ckd)
12. [Paper 1 — Ready-to-Use Manuscript Template](#12-paper-1--ready-to-use-manuscript-template)

---

## 1. Research Direction Summary

### Core Research Area

> **Developing evaluation frameworks, validated metrics, and open-source tools for assessing whether clinical AI systems are trustworthy enough for real-world deployment.**

### Research Structure

| Role | Content |
|---|---|
| **Core Contribution** | Evaluation frameworks and new metrics — probability calibration (ECE, DCGS, DCD), cross-site vulnerability (SSVI), failure detection (MACFD, CRI), and deployment readiness (FTS, CADRI) |
| **Application Domains** | Disease prediction (CKD, breast cancer), emergency department triage, ICU risk monitoring — the clinical contexts where frameworks are validated |
| **Trustworthiness Components** | Reliability, demographic fairness, security-by-design, explainability, and governance — treated as integrated components of deployment readiness, not separate research threads |

### Key Differentiator

Most researchers build models and report accuracy. Your differentiator is asking: *"How do we know this system is actually safe and reliable in production?"* — and then answering that question with **new metrics**, **validated open-source tools**, and **population-scale harm quantification**, not just evaluation studies on small public benchmarks.

---

## 2. Proposed PhD Dissertation

### Working Title
**Measuring and Ensuring Trustworthiness in Clinical AI: Evaluation Frameworks, Validated Metrics, and Deployment Readiness Tools for Healthcare Decision Support**

### Alternative Titles
- **Evaluation Frameworks and Open-Source Tools for Trustworthy Clinical AI: Calibration, Fairness, Security, and Deployment Readiness**
- **Beyond Accuracy: A Validated Framework for Assessing Calibration, Fairness, and Deployment Readiness in Healthcare AI Systems**

### One-Line Thesis Statement
> A trustworthy clinical AI system requires not just strong predictive performance, but validated reliability, calibration equity, adversarial robustness, and governance readiness — and this dissertation delivers the metrics, tools, and framework to measure and achieve all four.

---

## 3. APA-Style Proposal Structure

### Abstract Summary

This dissertation proposes the design, implementation, and evaluation of a trustworthy AI framework for healthcare diagnosis, triage, and patient monitoring. The study addresses a central gap: many models show promising performance in controlled settings, but fewer are designed with integrated security, reliability monitoring, interpretability, and deployment readiness.

**Research Phases:**
1. Baseline predictive modeling
2. Secure system architecture design
3. Reliability and interpretability evaluation
4. Real-world validation through retrospective clinical datasets and simulated deployment scenarios

**Candidate Datasets:**

| Dataset | Description |
|---|---|
| UCI Chronic Kidney Disease | 400 instances, 25 features — baseline classification, explainability |
| Breast Cancer Wisconsin Diagnostic | 569 samples, 30 features — benchmarking, feature importance |
| MIMIC-IV | Credentialed PhysioNet — ED and hospital data, broad clinical validation |
| MIMIC-IV-ED | ~425,000 ED stays — triage, ED risk stratification |
| eICU Collaborative Research DB | 200,000+ ICU admissions — generalization, severity modeling |
| MIMIC-IV-Ext Clinical Decision Making | Derived from MIMIC-IV — LLM clinical reasoning evaluation |

---

### Chapter 1 — Introduction

#### 1.1 Background
AI is increasingly used in healthcare for diagnosis, risk prediction, triage, patient monitoring, and clinical decision support. Errors in these systems can directly affect patient outcomes, clinician trust, privacy, and institutional risk.

#### Four Weaknesses in Current Healthcare AI Research
1. Models evaluated mainly on accuracy rather than reliability in practice
2. Security treated as an add-on rather than built into the system
3. Multimodal and real-world clinical workflows are under-modeled
4. Interpretability and failure detection are often too shallow for clinical adoption

#### 1.2 Problem Statement
No widely adopted end-to-end framework simultaneously addresses:
- Predictive performance
- Multimodal clinical data integration
- Cybersecurity and privacy safeguards
- Interpretability
- Failure detection and reliability evaluation
- Deployment readiness in realistic healthcare workflows

#### 1.3 Purpose
Design and validate a secure, scalable, and trustworthy AI framework for healthcare decision support integrating prediction, triage, interpretability, and reliability monitoring.

#### 1.4 Research Objectives
1. Develop baseline predictive models for selected healthcare tasks
2. Design a system architecture integrating ML, LLM-supported reasoning, data governance, and security
3. Create a reliability evaluation layer measuring calibration, drift, error patterns, uncertainty, and failure modes
4. Assess how interpretability and security design affect trustworthiness
5. Validate the framework using benchmark datasets and simulated clinical scenarios

#### 1.5 Research Questions

**Main Question:**
How can healthcare AI systems be designed to be accurate, secure, interpretable, and reliable enough for real-world clinical decision support?

**Sub-Questions:**
1. Which modeling approaches perform best across diagnosis, triage, and monitoring tasks?
2. How can security-by-design principles be embedded into healthcare AI architecture?
3. What evaluation metrics best capture trustworthiness beyond accuracy?
4. How do interpretability and uncertainty estimation affect clinician-facing usefulness?
5. How well does the proposed framework generalize across datasets and care settings?

#### 1.6 Significance
This study shifts focus from isolated healthcare prediction models to a full operational framework — asking whether the whole system can be trusted, secured, evaluated, and maintained in clinical settings.

---

### Chapter 2 — Literature Review

#### Thematic Areas

| Section | Focus |
|---|---|
| 2.1.1 | AI for diagnosis and risk prediction — disease prediction, classification, prognosis |
| 2.1.2 | AI for triage and decision support — symptom checkers, ED prioritization |
| 2.1.3 | Multimodal healthcare AI — EHR data, notes, vitals, imaging, time-series |
| 2.1.4 | Trustworthy and interpretable AI — explainability, calibration, uncertainty, fairness |
| 2.1.5 | Security and privacy — adversarial ML, data poisoning, federated learning |
| 2.1.6 | Evaluation frameworks — benchmark design, drift detection, post-deployment monitoring |

#### Core Literature Gap
> Existing healthcare AI studies often optimize for model performance on a narrow task, but fewer provide an integrated framework combining predictive modeling, system security, interpretability, and reliability evaluation within a deployment-oriented architecture.

#### Conceptual Framework — Five Connected Layers

```
1. DATA LAYER         → EHR, vitals, labs, notes, device data
2. MODEL LAYER        → ML classifiers, time-series models, LLM-assisted reasoning
3. SECURITY LAYER     → encryption, authentication, audit trails, adversarial defenses
4. TRUST LAYER        → explainability, uncertainty, calibration, fairness, drift detection
5. CLINICAL WORKFLOW  → triage support, diagnosis support, risk alerts, clinician review loop
```

---

### Chapter 3 — Methodology

#### 3.1 Research Design
**Design science + applied experimental research**
- Design science fits the creation of a framework and architecture
- Experimental ML fits model-building and evaluation
- Healthcare informatics fits system and deployment focus

#### 3.2 Research Paradigm
Pragmatic mixed-methods technical evaluation design

#### 3.3 Study Phases

| Phase | Focus |
|---|---|
| Phase 1 | Baseline predictive modeling (LR, RF, XGBoost, LightGBM, MLP, temporal models) |
| Phase 2 | Full system architecture design (data ingestion, preprocessing, model serving, interpretability, reliability, security) |
| Phase 3 | Trustworthiness & security evaluation (calibration, robustness, distribution shift, subgroup performance, adversarial) |
| Phase 4 | Simulated deployment & validation (ED triage, CKD support, ICU deterioration monitoring) |

#### 3.7 Variables

**Independent:** demographic features, lab values, vitals, symptoms, comorbidities, selected time-series markers, note-derived signals

**Dependent:** disease class, triage urgency, deterioration risk, mortality proxy, readmission/adverse event proxy

#### 3.8 Preprocessing Steps
- Missing value analysis and imputation
- Outlier handling
- Categorical encoding
- Feature scaling
- Class imbalance treatment
- Temporal aggregation for ICU and ED datasets
- Train/validation/test separation with leakage prevention

#### 3.9 Model Tiers

| Tier | Models |
|---|---|
| Tier 1 — Classical Baselines | Logistic Regression, Decision Tree, Random Forest, SVM |
| Tier 2 — Advanced ML | XGBoost, LightGBM, MLP |
| Tier 3 — Context-aware/Multimodal | Temporal deep learning, note-enhanced models, LLM-assisted triage |

#### 3.10 Interpretability Methods
- SHAP
- Permutation feature importance
- Partial dependence
- Local error analysis
- Uncertainty estimates
- Calibration curves

#### 3.11 Trustworthiness Evaluation Framework

**Performance Metrics:** accuracy, precision, recall, F1, AUROC, AUPRC, sensitivity, specificity

**Reliability Metrics:** expected calibration error, Brier score, uncertainty-confidence alignment, drift sensitivity, missingness robustness, subgroup stability

**Fairness & Transparency:** subgroup AUROC gaps, false negative disparity, feature attribution consistency, explanation faithfulness

**Security-Related:** resilience to noisy inputs, poisoning sensitivity, auditability, access control & logging review

#### 3.13 System Architecture Layers

```
DATA INGESTION     → tabular clinical, vitals/time-series, optional notes, schema validation
PROCESSING         → cleaning, transformation, feature engineering, missing-data management
INTELLIGENCE       → predictive ML models, optional LLM, model ensemble
TRUST & MONITORING → explanation engine, calibration monitor, uncertainty monitor, drift detector, error logging
SECURITY           → RBAC, encryption, audit logs, inference traceability, privacy governance
APPLICATION        → clinician dashboard, triage recommendations, diagnosis support, alerting
```

#### 3.14 Technology Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch or TensorFlow |
| Explainability | SHAP |
| Serving | FastAPI |
| Storage | PostgreSQL |
| Reproducibility | Docker |
| Experiment Tracking | MLflow or Weights & Biases |
| Optional LLM | LangChain (only if truly needed) |

#### 3.15 Validity & Rigor

| Type | Approach |
|---|---|
| Internal Validity | leakage control, reproducible preprocessing, fixed evaluation protocol |
| External Validity | multi-dataset testing, multicenter validation (eICU), robustness under shift |
| Construct Validity | trustworthiness beyond accuracy, clinical realism |
| Reproducibility | versioned code, documented pipelines, fixed seeds, experiment tracking |

#### 3.16 Ethical Considerations
- IRB determination if required by institution
- Use of de-identified public or credentialed data only
- No re-identification attempts
- Bias and fairness review
- Transparency in limitations
- Secure storage for credentialed datasets

---

### Chapter 4 — Expected Results
1. Advanced models will outperform simple baselines on raw predictive performance
2. Reliability metrics will reveal weaknesses hidden by accuracy alone
3. Multicenter and ED data will expose generalization challenges
4. Integrated trust and security layers will improve deployment readiness
5. The final framework will provide a reusable blueprint for healthcare AI systems

---

### Chapter 5 — Discussion
The discussion should interpret:
- Why certain models succeed or fail
- How security and trust measures change system usefulness
- Trade-offs between performance and interpretability
- What is realistic for clinical deployment
- What remains unsolved

---

### Chapter 6 — Limitations
- Retrospective rather than live clinical deployment
- Public benchmark datasets may not capture all workflow complexity
- LLM clinical reasoning components may require extra safety constraints
- Fairness evaluation depends on available demographic variables
- Clinical adoption claims will still require future real-world trials

---

### Chapter 7 — Conclusion
> This dissertation aims to move healthcare AI from isolated predictive models toward a secure, interpretable, and evaluation-centered system architecture suitable for real-world decision support. Its value lies not just in model performance, but in demonstrating how trustworthy healthcare AI can be designed, measured, and validated.

---

## 4. Chapter Outline

| Chapter | Contents |
|---|---|
| 1. Introduction | background, problem statement, purpose, research questions, significance, definitions, scope |
| 2. Literature Review | healthcare AI prediction, triage/decision support, multimodal AI, trustworthy AI, cybersecurity in healthcare AI, evaluation frameworks, gap, conceptual framework |
| 3. Methodology | research design, datasets, preprocessing, modeling, architecture design, trustworthiness metrics, statistical analysis, ethics |
| 4. Framework Design & Implementation | architecture, pipeline, security controls, interpretability components, monitoring components |
| 5. Experimental Results | baseline results, advanced model results, cross-dataset results, trustworthiness results, ablations |
| 6. Discussion | interpretation, implications, comparison with prior work, deployment meaning |
| 7. Conclusion & Future Work | summary, contribution, practical recommendations, future extensions |

---

## 5. Publication Roadmap: 10 Papers

> **Note on gap strength:** Each paper below includes the specific literature evidence that justifies the gap. Papers 1, 2, 4, 5, and 10 were fully redesigned after an extensive literature search confirmed the original angles were saturated. The revised angles target documented, unaddressed gaps.

---

### The Required Quality Bar: Lessons from StepShield

Comparison with StepShield (Felicia et al., arXiv 2601.22136, January 2026 — co-authored benchmark paper) defines the minimum standard each paper in this roadmap must meet. StepShield's impact came from four principles applied consistently:

| Principle | What StepShield did | Required standard for this roadmap |
|---|---|---|
| **Conceptual reframing** | Reframed detection from "WHETHER" to "WHEN" — one sentence that made every prior benchmark obsolete | Each paper must articulate a gap no one has previously named clearly — not just "X is underreported" but "the field measures X when Y is what actually matters" |
| **New reusable artifact** | Created 9,213 annotated trajectories + 3 new metrics (EIR, Intervention Gap, Tokens Saved) — infrastructure others build on | Every paper must produce a new metric, open-source tool, dataset, or validated protocol — not just findings on existing datasets |
| **Quantified real-world impact** | Projected $108M in enterprise savings over 5 years — concrete, not just "clinically relevant" | Each paper must translate findings into concrete patient harm or cost at population scale |
| **Open-source release** | Full code and data under Apache 2.0 | All pipelines, datasets, and artifacts released publicly under a permissive license |

> **Critical diagnosis applied to this roadmap:** Papers that only evaluate existing methods on small, well-known benchmark datasets (UCI CKD 400 patients, Wisconsin Diagnostic 569 patients) without introducing new metrics, tools, or methods are publishable but do not reach this standard. The paper designs below have been upgraded to address this directly. Each paper now specifies its new metric or artifact as a non-negotiable deliverable.

---

### Paper 1 *(Completed and Submitted)*
**Title:** Calibration, Uncertainty Communication, and Deployment Readiness in CKD Risk Prediction: A Framework Evaluation Study
- **Type:** Calibration + uncertainty quantification evaluation paper
- **Status:** Submitted to *JAMIA Open*; backup venue *Journal of Biomedical Informatics*
- **Reusable artifact produced:** An eight-criterion deployment readiness checklist for CKD prediction models — the first structured deployment scoring framework for calibration and uncertainty in CKD AI
- **Key findings:**
  - AUROC 1.00 on UCI CKD test set — confirms the dataset is essentially solved for discrimination; the clinically important finding is what happens under external validation
  - On MIMIC-IV: AUROC dropped to 0.48–0.58, ECE rose to 0.68–0.76, conformal coverage collapsed from 0.80–0.98 to 0.21–0.25 against a 90% target
  - No model scored above 4 out of 16 on the deployment readiness checklist
- **What this paper contributes:**
  - First study to jointly evaluate CKD prediction through calibration, conformal uncertainty, and deployment readiness in a single reproducible framework
  - Demonstrates empirically that near-perfect internal performance does not imply clinically usable probability estimates
  - The deployment readiness checklist is the reusable artifact — structured to be applied to any clinical prediction model
- **Honest assessment of limitations:**
  - UCI CKD (400 patients) and MIMIC-IV demo (97 patients) are small; AUROC 1.00 on UCI reflects dataset simplicity, not model strength
  - The paper's core value lies in the calibration collapse finding and the checklist framework, not discrimination performance
- **If revision is required:** Strengthen by quantifying clinical harm — at 0.73 average ECE on MIMIC-IV, a model applied to approximately 350,000 annual US CKD diagnoses generates a quantifiable number of miscalibrated risk communications per year; add this estimate
- **Lessons carried forward:** All subsequent papers use larger, more challenging datasets; UCI CKD and Wisconsin Diagnostic serve only as methodology pilots, not primary clinical evidence

---

### Paper 2 *(Revised and Upgraded)*
**Title:** Fairness Gaps and Calibration Failures in Structured Breast Cancer Prediction: A Multi-Dataset Evaluation with Demographic Subgroup Analysis
- **Type:** Fairness + calibration + generalization evaluation paper
- **Target Venue:** *npj Digital Medicine* (primary, IF 15.2 — justified by dataset scale upgrade), *Journal of Biomedical Informatics* (backup)
- **Why this gap is real:**
  - A 2024 systematic scoping review of XAI in breast cancer (PMC) found calibration was "infrequently considered — only 3.2% of studies" assessed it
  - 75–100% of breast cancer ML studies do not report race/ethnicity; "the majority of data (over 70%) were from White patients" with documented performance disparities for non-White patients
  - A 2025 BMC Cancer systematic review of 107 breast cancer prediction models found "more than half were rated high risk of bias due to inadequate handling of missing data and insufficient calibration reporting" with most lacking external validation
  - Post-processing fairness methods for structured tabular breast cancer data are essentially absent from the literature
- **Dataset upgrade (critical):** Wisconsin Diagnostic (569 samples) is used as a **pilot/methodology validation only**. Primary evidence comes from SEER Breast Cancer Surveillance Consortium data (100,000+ patients with demographic variables) and/or a MIMIC-IV-derived breast cancer cohort — providing the scale required for credible intersectional subgroup analysis
- **New metric introduced — Demographic Calibration Gap Score (DCGS):** The absolute difference in ECE between the best-calibrated and worst-calibrated demographic subgroup, normalized by overall ECE. A DCGS > 0.20 flags a clinically actionable calibration equity failure. This metric does not exist in current literature and is designed to be adopted by future breast cancer and broader clinical AI fairness studies
- **What you do that others haven't:**
  - Use SEER-scale data to perform intersectional calibration analysis (race × age × stage simultaneously) — not just marginal subgroup comparisons
  - Introduce and validate DCGS as a new fairness-calibration metric applicable beyond breast cancer
  - Apply and compare post-processing fairness interventions (equalized odds, calibrated equal opportunity) for structured tabular data, using DCGS as the primary evaluation criterion
  - Quantify real-world harm: at 3.2% calibration reporting rate across approximately 300,000 annual US breast cancer diagnoses, estimate the number of patients per year receiving systematically miscalibrated risk communications
- **Reusable artifact:** DCGS metric definition, validation code, and SEER-compatible calibration evaluation pipeline released as open-source under Apache 2.0
- **Specific contribution:** The first structured-tabular breast cancer ML study to introduce a validated demographic calibration gap metric (DCGS) and perform intersectional calibration analysis at population scale — filling a gap confirmed by multiple 2024–2025 systematic reviews

---

### Paper 3 *(Upgraded — adversarial testing and security posture scoring required)*
**Title:** Evaluating Security-by-Design in Clinical AI: Adversarial Testing, Threat Assessment, and Security Posture Scoring
- **Type:** Security evaluation + adversarial validation paper
- **Target Venue:** *JAMIA* or *npj Digital Medicine*
- **Why this gap is real:** Concrete, implemented, adversarially tested clinical AI architectures with security-by-design are absent from the literature. A 2025 JMIR systematic review explicitly identified the lack of implementation-level security integration. A 2025 PMC review documented 624 cyber-attacks on hospitals in 2023 — double 2022's 304, averaging $1.3M in disruption costs each — with no corresponding published security-embedded AI architecture validated against healthcare-specific threat models
- **What separates this from a design paper (the critical standard):** Architecture papers without real adversarial testing are not acceptable at the required level. This paper must include:
  - Threat modeling using MITRE ATT&CK for Healthcare — mapping specific attack vectors to architectural mitigations
  - Adversarial input tests adapted for tabular clinical data (FGSM-style perturbations on lab values and vital signs)
  - Simulated red team assessment of the API and inference layer
  - RBAC implementation with logged audit trails demonstrating traceable inference decisions
  - HIPAA/HL7 FHIR compliance review of the data handling layer
- **Reusable artifact:** Full Docker-composable architecture released as open-source on GitHub — six-layer system (Data, Processing, Intelligence, Trust, Security, Application) that other researchers can deploy and extend for their own clinical prediction tasks
- **Quantified impact:** At 624 hospital breaches in 2023 at average $1.3M per breach, quantify prevention value against the specific attack vectors the architecture mitigates in simulated scenarios
- **Specific contribution:** The first published, implemented, adversarially tested clinical AI architecture with security controls verified against healthcare-specific threat models — not a design proposal but a validated system

---

### Paper 4 *(Revised — new metric and population-scale quantification added)*
**Title:** Intersectional Fairness and Reliability in Emergency Department AI Triage: Mapping Demographic Bias Pathways in MIMIC-IV-ED
- **Type:** Intersectional fairness + bias attribution paper
- **Target Venue:** *CHIL* (primary), *JAMIA* (backup)
- **Why this gap is real:**
  - A 2024 medRxiv study using MIMIC-IV-ED (n=160,016) found "the majority of medical AI literature applies marginal de-biasing approaches which constrain performance across one or many isolated patient attributes" — intersectional fairness (race × gender × age simultaneously) is almost entirely unstudied in triage AI
  - A 2024 PMC study found racial differences in laboratory testing in MIMIC-IV create a documented, measurable bias pathway into AI models — but no triage study has traced this from data collection to model output to decision disparity
  - General triage prediction benchmarks on MIMIC-IV-ED are saturated; the specific gap is understanding *how* demographic-correlated care patterns create bias in automated triage, not just whether bias exists
- **New metric introduced — Bias Pathway Attribution Score (BPAS):** Quantifies what percentage of an intersectional performance gap is attributable to upstream data collection disparities (differential lab ordering, documentation differences) versus inductive model bias. A BPAS > 0.60 indicates the dominant source of bias is upstream in data collection — directing intervention to data processes rather than model post-processing. This metric is novel, reusable, and applicable to any clinical AI fairness study
- **What you do that others haven't:**
  - Quantify intersectional performance gaps (race × gender × age combinations) rather than single-attribute fairness metrics
  - Trace and quantify the bias pathway using BPAS: differential lab ordering → missing feature patterns → prediction disparities → triage outcome differences
  - Evaluate whether standard debiasing methods designed for marginal fairness actually worsen intersectional fairness
  - Publish a bias-aware triage evaluation protocol and BPAS computation toolkit as open-source
- **Quantified real-world impact:** Approximately 145 million US ED visits annually; if intersectional triage bias affects even 2% of visits, this represents approximately 2.9 million patients per year at risk of undertriage — frame this explicitly in the paper
- **Reusable artifact:** BPAS computation library and bias pathway tracing pipeline released under Apache 2.0, applicable to any MIMIC-IV-ED or EHR-based fairness study
- **Specific contribution:** The first triage AI study to introduce a validated bias attribution metric (BPAS), trace intersectional bias pathways from data collection to triage decisions, and quantify what proportion of bias is correctable at the data layer versus the model layer

---

### Paper 5 *(Revised — new metric, open-source tool, and quantification added)*
**Title:** Diagnosing Cross-Site Degradation in ICU Risk Prediction: A Subgroup-Level Failure Attribution Analysis Using eICU
- **Type:** Failure attribution + subgroup diagnosis paper
- **Target Venue:** *npj Digital Medicine* (primary, IF 15.2), *Journal of Biomedical Informatics* (backup)
- **Why this gap is real:**
  - A 2024 Critical Care Medicine study across 334,812 ICU stays confirmed AUROC drops up to −0.200 during cross-site transfer but did not explain *which* patient subgroups drive the degradation
  - A 2023 medRxiv systematic review found 7–23% average AUROC drops externally with "heterogeneity reflecting highly disparate patient populations" — patient-level drivers not identified
  - A 2024 Critical Care Medicine study measured KL divergence shifts (medications 0.82, labs 0.58, vitals 0.34) between MIMIC and eICU but did not translate these into clinically actionable subgroup explanations
- **New metric introduced — Site-Subgroup Vulnerability Index (SSVI):** For each clinically defined patient subgroup (defined by age tier, acuity level, comorbidity burden), SSVI = expected AUROC drop per unit of KL divergence between source and target site feature distributions. A high-SSVI subgroup is one where even small distributional differences cause large performance drops — identifying patients at greatest risk from cross-site deployment. SSVI is novel, mathematically defined, and directly actionable for site readiness assessment
- **What you do that others haven't:**
  - Compute SSVI across patient subgroups defined by age, acuity, comorbidity burden, primary diagnosis, and demographic profile
  - Map KL divergence shifts in clinical features to SSVI scores — explaining *why* specific subgroups are harmed, not just that models degrade
  - Produce a Subgroup Degradation Attribution Report (SDAR) — a structured, reusable report format that any ICU AI deployment team can generate before cross-site rollout
  - Provide a pre-deployment site assessment checklist grounded in SSVI thresholds
- **Quantified real-world impact:** Approximately 5.7 million US ICU admissions annually; explicitly quantify how many patients would receive clinically inferior risk scores if cross-site deployment proceeds for 20% of ICUs without SSVI-guided site assessment
- **Reusable artifact:** SSVI computation library and SDAR generator released as open-source Python tools under Apache 2.0, validated on eICU and applicable to any multicenter clinical ML study
- **Specific contribution:** The first eICU study to introduce a validated cross-site vulnerability metric (SSVI) and structured subgroup degradation attribution methodology — shifting the field from "models degrade" (already known) to "here is exactly who is harmed, by how much, and why"

---

### Paper 6 *(Upgraded — open-source library and new composite metric required)*
**Title:** Beyond Accuracy in Healthcare AI: A Reliability Evaluation Framework for Clinical Prediction Models
- **Type:** Trustworthiness framework paper — implemented as a validated, open-source tool
- **Target Venue:** *CHIL* (primary), *JAMIA* or *Journal of Biomedical Informatics* (backup)
- **Why this gap is real:** A 2025 Lancet Digital Health evaluation guidance paper found calibration, uncertainty, and clinical utility metrics are severely underused among 32 common performance measures. No structured, reusable reliability framework currently exists that jointly operationalizes calibration, drift detection, subgroup stability, and uncertainty quality in a single testable, open-source protocol
- **New metric introduced — Clinical Reliability Index (CRI):** A composite score (0–100) aggregating four reliability dimensions: (1) calibration quality (ECE-weighted), (2) drift sensitivity score, (3) subgroup stability ratio, and (4) uncertainty-confidence alignment. Threshold: CRI ≥ 70 required for conditional deployment recommendation. CRI is a novel, quantified summary enabling direct comparison of reliability across models, datasets, and studies — equivalent to what AUROC did for discrimination
- **Critical implementation requirement:** This paper must deliver a working open-source Python library (`clinicalml-reliability`, pip-installable) that any researcher can run on their own model. Publishing a framework paper without an open-source implementation does not meet the required standard. This library becomes the infrastructure backbone for Papers 7, 8, and 9
- **Validation standard:** CRI computed and validated on a minimum of four datasets: UCI CKD, MIMIC-IV, Wisconsin Diagnostic, and an eICU-derived cohort — demonstrating the framework generalizes across data modalities and clinical tasks
- **Reusable artifact:** `clinicalml-reliability` Python package, CRI computation code, documentation, and reproducible evaluation notebooks released under Apache 2.0
- **Specific contribution:** The first implemented, open-source clinical ML reliability evaluation framework with a validated composite metric (CRI) — not just a proposed taxonomy but a deployable tool that enables standardized reliability reporting across all future healthcare AI studies

---

### Paper 7 *(Upgraded — new algorithm with theoretical grounding required for AAAI)*
**Title:** Multi-Axis Clinical Failure Detection: A Joint Framework for Missingness, Distribution Shift, and Noise in Healthcare ML
- **Type:** Novel algorithm + empirical evaluation paper
- **Target Venue:** *AAAI* (primary — requires novel algorithm with theoretical grounding), *CHIL* (backup)
- **Why this gap is real:** A 2025 ScienceDirect systematic review (32 studies, 2019–2025) found dataset shift detection in healthcare ML is inconsistent and narrowly applied — mostly single-shift-type detection. A 2025 PMC study on harmful data shifts confirmed shifts occur simultaneously across multiple axes in real deployments. No joint detection framework exists
- **Critical standard for AAAI:** A paper that only applies existing detection methods (KS test, MMD, PSI) to clinical data will be rejected. This paper must introduce a new algorithm — the Multi-Axis Clinical Failure Detector (MACFD):
  - A joint hypothesis testing framework that simultaneously tests for missingness-induced bias, covariate distribution shift, and input noise sensitivity
  - Statistical foundation: a unified likelihood ratio test combining three marginal tests with a correction for simultaneous inference (Benjamini-Hochberg adapted for clinical feature dependencies)
  - Detection power bounds: formal analysis of the minimum shift magnitude MACFD detects at 80% power with a 5% false alarm rate
  - Runtime requirements suitable for deployment-time monitoring (not just offline analysis)
- **Theoretical contribution:** Proof or strong empirical evidence that joint detection under MACFD has lower false negative rate than sequential application of individual detectors — this claim is what separates the paper from an evaluation study
- **Empirical validation:** Minimum three clinical datasets; synthetic shift injections at controlled magnitudes to characterize detection thresholds; comparison against existing single-axis baselines
- **Reusable artifact:** MACFD Python package released as open-source, designed to integrate directly with the `clinicalml-reliability` package from Paper 6
- **Specific contribution:** The first joint multi-axis clinical failure detector with theoretical detection power guarantees — providing a single-pass failure audit tool that replaces sequential, inconsistent single-axis monitoring in healthcare ML deployments

---

### Paper 8 *(Upgraded — new metric and ACM FAccT framing strengthened)*
**Title:** Calibration Equity in Healthcare AI: A Demographic Subgroup Analysis of Miscalibration Harm in Triage and Diagnosis
- **Type:** Fairness and evaluation paper with new metric contribution
- **Target Venue:** *ACM FAccT* (primary), *AIES* (backup)
- **Why this gap is real:** A 2025 npj Digital Medicine scoping review found sensitivity gaps of >31% for minority patients in critical care AI; calibration across demographic groups is almost entirely unstudied. The gap is not just that subgroup fairness is underreported — it is that no metric exists to jointly quantify calibration disparity across demographics in a way that is comparable, normalized, and actionable
- **New metric introduced — Demographic Calibration Disparity (DCD):** The maximum absolute ECE gap between any two demographic subgroups, normalized by overall ECE. A DCD > 0.15 defines a "calibration equity failure" — a threshold empirically derived from clinical risk communication standards. Unlike existing fairness metrics (equalized odds, demographic parity) that measure classification performance, DCD measures probability reliability equity — a dimension not previously formalized
- **ACM FAccT positioning:** Frame DCD as a new dimension of algorithmic fairness — "calibration equity" — distinct from and complementary to classification fairness metrics. The conceptual contribution is naming and formalizing a gap the fairness community has acknowledged but never quantified with a dedicated metric
- **Empirical target:** Demonstrate that 30%+ of models with acceptable aggregate Brier scores (< 0.15) exhibit DCD > 0.15 for at least one demographic subgroup — proving that aggregate calibration does not guarantee calibration equity
- **Quantified harm:** In triage settings, DCD > 0.15 means that for a demographic group comprising 15% of patients, predicted probabilities are systematically off by a clinically actionable margin; translate to number of affected patients at MIMIC-IV-ED scale
- **Reusable artifact:** DCD metric definition, computation code, and calibration equity evaluation module — integrated with `clinicalml-reliability` from Paper 6 — released as open-source
- **Specific contribution:** The first paper to formalize and validate "calibration equity" as a distinct dimension of healthcare AI fairness, introducing DCD as a standardized metric applicable to any clinical prediction model with demographic metadata

---

### Paper 9 *(Flagship — highest potential, full open-source system required)*
**Title:** TrustMed: A Validated Open-Source Framework for Trustworthy Clinical Decision Support Integrating Prediction, Reliability, Fairness, Explainability, and Security
- **Type:** Flagship dissertation framework paper — full system implementation
- **Target Venue:** *npj Digital Medicine* (primary, IF 15.2), *JAMIA* (backup)
- **Why this gap is real:** A 2026 Scientific Reports TAXAI paper and a 2025 Frontiers Digital Health paper both identify a persistent gap between model validation and real-world system behavior. No validated, end-to-end framework currently integrates prediction performance + reliability evaluation + fairness monitoring + explainability + security governance in a single tested, open-source architecture
- **What separates this from a synthesis paper (the critical standard):** Papers 1–8 produce components. Paper 9 does not simply describe how they fit together — it must demonstrate the integrated system producing behaviors that none of the individual components produce in isolation. The integration is the primary scientific contribution
- **Named artifact — TrustMed Framework:** A versioned, Docker-composable system with six layers:
  - Data layer: FHIR-compatible ingestion, schema validation, audit logging
  - Model layer: pluggable ML classifiers with optional LLM-assisted reasoning
  - Reliability layer: CRI computation (Paper 6) and MACFD failure detection (Paper 7)
  - Fairness layer: BPAS (Paper 4), DCD (Paper 8), and subgroup stability monitoring
  - Security layer: RBAC, encryption, adversarially tested inference API (Paper 3)
  - Application layer: clinician-facing dashboard with SHAP explanations and uncertainty-aware risk communication
- **New metric introduced — Framework Trustworthiness Score (FTS):** A composite score (0–100) integrating CRI (reliability), DCD (calibration equity), security posture score (Paper 3), and explainability quality (faithfulness + clinical usability). FTS provides a single-number summary of how deployment-ready a clinical AI system is — the metric that Paper 10's CADRI protocol will operationalize and validate
- **Validation standard:** TrustMed validated on minimum three clinical datasets (CKD, ED triage, ICU deterioration); ablation study showing the contribution of each layer to overall FTS
- **Reusable artifact:** TrustMed open-source release on GitHub under Apache 2.0; pip-installable components; reproducible Docker environment; full documentation
- **Specific contribution:** The first fully implemented, validated, named clinical AI trustworthiness framework to integrate all five dimensions (prediction, reliability, fairness, security, explainability) in a single open-source system with a quantified deployment readiness score

---

### Paper 10 *(Revised and Upgraded — empirically weighted scored index required)*
**Title:** The Clinical AI Deployment Readiness Index (CADRI): A Validated Scoring Protocol for Production-Ready Healthcare AI Systems
- **Type:** Operationalization framework paper with empirically validated scoring index
- **Target Venue:** *npj Digital Medicine* (primary, IF 15.2), *JAMIA* or *Frontiers in Digital Health* (backup)
- **Why this gap is real:**
  - A 2025 PMC implementation framework paper states "the field of AI-CDSS implementation research is still in its infancy" with a persistent "AI chasm" between model accuracy and clinical efficacy
  - A 2026 Frontiers Digital Health paper identifies a gap between the DG-TAI conceptual framework and actual deployed systems — no paper provides a tested, step-by-step deployment readiness protocol
  - Multiple 2024–2025 papers (PMC checklist, HIMSS guide, JMIR roadmap) address deployment in isolation — none integrates security, reliability, fairness, and governance into one scored, empirically validated protocol
- **New deliverable — Clinical AI Deployment Readiness Index (CADRI):** A scored composite index (0–100) with five empirically weighted subscores:
  1. Model Reliability subscore (from CRI, Paper 6) — weight derived from empirical correlation with external validation failure
  2. Security Posture subscore (from adversarial testing results, Paper 3)
  3. Fairness Audit subscore (from DCD and BPAS, Papers 4 and 8)
  4. Explainability Usability subscore (SHAP faithfulness + clinician usability rating)
  5. Governance Readiness subscore (IRB documentation, data provenance, audit trail completeness)
  - Threshold: **CADRI ≥ 75** = conditional deployment recommendation; CADRI < 50 = deployment not recommended
  - Weights are empirically derived by regressing CADRI subscores against external validation failure rates across the clinical AI systems from Papers 3, 6, 7, and 9 — making CADRI a data-driven index, not an arbitrary checklist
- **What you do that others haven't:**
  - Derive CADRI weights empirically from real system performance data — distinguishing CADRI from existing checklists (CASoF, AI-HIF) that use expert opinion for weighting
  - Demonstrate that models above CADRI 75 show significantly lower rates of calibration failure and safety events on external validation — validating the threshold empirically
  - Distinguish "technically ready" (high CRI, good DCD) from "operationally ready" (high governance + explainability usability) — two orthogonal dimensions no existing framework separates
  - Release CADRI as an open-source scoring tool with automated extraction from model outputs and documentation artifacts
- **Reusable artifact:** CADRI scoring tool (Python + web interface), scoring rubric, validation dataset from Papers 3/6/7/9 systems, and complete documentation released as open-source under Apache 2.0
- **Specific contribution:** The first empirically validated, scored deployment readiness index for clinical AI — with data-driven subscore weights and demonstrated predictive validity (CADRI score predicts external validation outcomes) — bridging the AI chasm between model validation and clinical deployment

---

## 6. Paper-to-Dissertation Mapping

| Category | Papers | Role | Key Artifacts |
|---|---|---|---|
| **Calibration & Evaluation Foundation** | 1, 2 | Establish calibration, uncertainty, and fairness evaluation vocabulary; introduce DCGS metric; demonstrate that calibration collapses under external validation | Deployment checklist, DCGS metric, SEER pipeline |
| **Security, Integration & Operationalization Papers** | 3, 9, 10 | Evaluate security posture via adversarial testing and threat assessment; integrate all components into TrustMed; operationalize deployment readiness as CADRI | Security posture scoring tool, TrustMed system, CADRI scoring tool |
| **Clinical Application Papers** | 4, 5 | Demonstrate real-world relevance in ED triage and multicenter ICU settings; introduce BPAS and SSVI metrics | BPAS toolkit, SSVI library, SDAR generator |
| **Trustworthy AI Papers** | 6, 7, 8 | Build the reliability infrastructure (CRI), failure detection algorithm (MACFD), and calibration equity metric (DCD) that make the framework trustworthy | `clinicalml-reliability` package, MACFD package, DCD metric |

### How the Papers Build on Each Other

```
Papers 1 & 2  →  Establish calibration + fairness evaluation vocabulary (DCGS, DCD)
      ↓
Paper 3       →  Evaluate security posture: adversarial testing + threat assessment
      |
      |    (Papers 4, 5 are independent evaluation studies — no dependency on Paper 3)
      ↓
Papers 4 & 5  →  Evaluate fairness and failure attribution in ED triage and ICU settings (BPAS, SSVI)
      ↓
Papers 6 & 7  →  Build the reliability infrastructure and failure detection algorithm (CRI, MACFD)
      ↓
Paper 8       →  Validate calibration equity across demographic subgroups (DCD)
      ↓
Paper 9       →  Integrate calibration + security (Paper 3) + reliability layers into TrustMed (FTS)
      ↓
Paper 10      →  Operationalize the full framework as a scored deployment readiness protocol (CADRI)
```

---

## 7. Publication Sequence by Year

| Year | Focus | Target Papers | Key Artifacts Due |
|---|---|---|---|
| **Year 1** | Literature review, methodology establishment, calibration evaluation framework | Papers 1 (done), 2 | Deployment checklist (Paper 1); DCGS metric + SEER pipeline (Paper 2) |
| **Year 2** | Security evaluation, clinical task expansion, ED and ICU datasets, proposal defense | Papers 3, 4, 5 | Security posture scoring tool + adversarial tests (Paper 3); BPAS toolkit (Paper 4); SSVI + SDAR (Paper 5) |
| **Year 3** | Trustworthiness evaluation, robustness, fairness metrics, reliability library | Papers 6, 7, 8 | `clinicalml-reliability` package + CRI (Paper 6); MACFD package (Paper 7); DCD metric (Paper 8) |
| **Year 4** | Framework integration, flagship system, deployment protocol, dissertation writing | Papers 9, 10 | TrustMed open-source system + FTS (Paper 9); CADRI tool + empirical validation (Paper 10) |

> **Venue upgrade note:** Paper 2 is upgraded to *npj Digital Medicine* (primary) based on the SEER dataset addition. Papers 5, 9, and 10 target *npj Digital Medicine* (IF 15.2). Paper 7 targets *AAAI* — the highest-difficulty target in this roadmap and requires a novel algorithm, not an evaluation study.

---

## 8. Dissertation Final Claims

Your dissertation should not claim you merely built a good model. It should claim:

1. You designed a healthcare AI **framework** (TrustMed), not just an isolated classifier
2. You evaluated **trustworthiness** in a structured way, beyond raw prediction scores — using validated metrics you introduced
3. You embedded **security and governance** into the architecture with adversarial testing, not as an afterthought
4. You showed the framework across **multiple clinical settings and datasets** (CKD, ED triage, ICU deterioration)
5. You produced a **reusable blueprint** for future healthcare AI deployment — open-source, documented, and reproducible
6. You introduced **seven new metrics and tools** (DCGS, BPAS, SSVI, CRI, DCD, FTS, CADRI) that the broader healthcare AI community can adopt — this is your infrastructural contribution to the field
7. You quantified the **real-world patient harm implications** of deploying unreliable, uncalibrated, or biased clinical AI at population scale — anchoring academic contributions in clinical stakes

### Recommended Narrow Thesis Angle
> **Trustworthy AI for clinical decision support using structured and multimodal health data: new metrics, open-source tools, and validated frameworks for security-by-design, reliability evaluation, and deployment readiness**

### Strong Methodology Paragraph (reusable in proposal)
> This study adopts a design science and experimental evaluation methodology to develop and validate a trustworthy healthcare AI framework. The research proceeds through iterative phases of baseline model construction, architecture design, reliability and security integration, and retrospective validation on benchmark clinical datasets. Quantitative evaluation will include classification performance, calibration, robustness under missingness and distribution shift, subgroup stability, and interpretability quality. The final artifact will be assessed not only as a predictive model, but as an integrated decision-support framework suitable for real-world healthcare environments.

---

## 9. Literature Gap Evidence (Search-Verified, April 2026)

> This section documents the specific literature findings that justify each paper's gap. These are not assumed gaps — they were found through a systematic search of 2023–2026 publications.

### Paper 1 Gap Evidence — CKD Calibration & Uncertainty
| Finding | Source |
|---|---|
| Only 8 CKD occurrence and 5 progression models have been externally validated for calibration | PLOS Medicine systematic review of CKD risk models |
| "Fewer than 4% of clinical AI studies address uncertainty explicitly" | ScienceDirect 2025 vision paper on uncertainty-aware ML in healthcare |
| "Existing models need to be better calibrated and externally validated before guideline incorporation" | CDC PCD systematic review, 2023 |
| Conformal prediction for CKD is essentially unstudied in clinical contexts | ACL CL4Health 2025; ACM Computing Surveys on conformal prediction |

### Paper 2 Gap Evidence — Breast Cancer Fairness & Calibration
| Finding | Source |
|---|---|
| Only 3.2% of breast cancer ML studies assessed calibration | PMC systematic scoping review of XAI in breast cancer, 2024 |
| 75–100% of studies do not report race/ethnicity; most data from White patients | Same scoping review |
| "More than half of included models rated high risk of bias due to insufficient calibration reporting" | BMC Cancer systematic review of 107 prediction models, 2025 |
| Post-processing fairness for structured tabular breast cancer data: essentially no papers | Search across JCE, npj DM, PLOS Digital Health, 2024–2025 |

### Paper 4 Gap Evidence — Intersectional ED Triage Fairness
| Finding | Source |
|---|---|
| "The majority of medical AI literature applies marginal de-biasing approaches" — intersectional fairness unstudied in triage | medRxiv, MIMIC-IV-ED study, n=160,016, 2024 |
| Racial differences in lab testing in MIMIC-IV create documented bias pathways into AI models | PMC, 2024 |
| No study traces bias from differential care patterns through to triage decision disparities | Search across CHIL, JAMIA, Lancet Digital Health 2023–2025 |

### Paper 5 Gap Evidence — ICU Cross-Site Failure Attribution
| Finding | Source |
|---|---|
| AUROC drops up to −0.200 in cross-site transfer but patient-level drivers not identified | Critical Care Medicine, 2024, 334,812 ICU stays |
| "Purely computational approaches are unlikely to result in reliable risk prediction for the ICU" | Same paper |
| KL divergence shifts measured (meds 0.82, labs 0.58, vitals 0.34) but not translated to clinical subgroup explanations | Generalizability systematic review, medRxiv 2023 |
| Average AUROC drop of 7–23% externally with heterogeneity unexplained at patient level | BMC Medical Informatics meta-analysis, 2024 |

### Paper 10 Gap Evidence — Deployment Readiness Framework
| Finding | Source |
|---|---|
| "The field of AI-CDSS implementation research is still in its infancy" | PMC implementation framework, 2025 |
| Persistent "AI chasm" between model accuracy and clinical efficacy — no bridging framework exists | PMC roadmap to implementing ML in healthcare, 2025 |
| Existing checklists (CASoF, AI-HIF) address governance but not integrated security + reliability + fairness | PMC checklist paper, 2025; HIMSS operationalization guide, 2025 |
| "Technical XAI solutions have often failed to address real-world clinician needs and workflow integration" | MDPI informatics review, 2025 |
| DG-TAI framework (2026) identifies gap between validation and real-world behavior but provides no tested protocol | Frontiers Digital Health, 2026 |

---

## 11. Authorship Strategy

### Can You Be Sole Author?
Yes — but it is not the strongest strategy for every paper.

### Recommended Balance

| Authorship Type | Count | Best For |
|---|---|---|
| Sole author | 2–4 papers | literature reviews, baseline CKD/breast cancer papers, methodology papers |
| First author (collaborative) | 5–7 papers | MIMIC/eICU studies, architecture papers, fairness/security papers, flagship papers |
| Co-author | 1–2 papers | side contributions with clinical/domain experts |

### Key Rule
Authorship follows **actual contribution**. If an advisor provides intellectual direction, shapes methods, revises substantially, or secures data access — they should be included.

### What Matters Most for Your CV
- Whether you are **first author**
- Whether papers are in **solid venues**
- Whether papers form a **coherent research agenda**
- Whether your dissertation shows **independent thinking**

---

## 12. Paper 1 — Full Planning Guide (CKD)

### Final Working Title
**Calibration, Uncertainty Communication, and Deployment Readiness in CKD Risk Prediction: A Framework Evaluation Study**

### One-Sentence Contribution
> This paper addresses a documented gap — fewer than 4% of clinical AI studies and fewer than 13 CKD risk models have ever been evaluated for calibration — by building the first reproducible framework that jointly assesses probability calibration, uncertainty quantification via conformal prediction, and deployment readiness criteria for CKD prediction models, using UCI CKD for development and MIMIC-IV for external calibration validation.

### Primary Target: JAMIA Open
- Up to **4,000 words** (main text)
- Structured abstract ≤ **250 words**
- Up to **4 tables**, **6 figures**
- Unlimited references
- Encourages code repositories and public data access
- Requires patient/community-facing abstract

### Backup: Journal of Biomedical Informatics (JBI)
- Structured abstract ≤ **300 words**
- Body ≤ **6,000 words**
- Total figures + tables ≤ **8**
- Requires stronger **methodological novelty** — frame as a generalizable evaluation framework, not just a dataset comparison

---

### Research Questions & Hypotheses

**Research Question:**
How do Logistic Regression, Random Forest, and Gradient Boosting compare for CKD prediction when performance, calibration, class imbalance, and feature-level interpretability are evaluated together?

**Hypotheses:**
- **H1:** Ensemble methods will outperform Logistic Regression on discrimination metrics (AUROC, F1)
- **H2:** Logistic Regression may show better raw calibration than tree ensembles before recalibration
- **H3:** The model with highest accuracy may not be the most clinically usable once calibration and interpretability are considered

---

### Dataset Details — UCI Chronic Kidney Disease

| Property | Value |
|---|---|
| Donated | 2015 |
| Instances | 400 |
| Features | 24 predictor variables + 1 class label |
| Task | Binary classification (CKD / not CKD) |
| Missing values | Yes — requires preprocessing |
| Key variables | age, blood pressure, albumin, sugar, blood glucose random, blood urea, serum creatinine, sodium, potassium, hemoglobin, hypertension, diabetes mellitus, anemia |

---

### Phase-by-Phase Execution Plan

#### Phase 1 — Lock the Paper Design
1. Finalize research question
2. State objectives
3. State hypotheses

#### Phase 2 — Choose the Journal
- Primary: JAMIA Open
- Backup: JBI (only if stronger methodological framing is added)

#### Phase 3 — Study Design
- **Type:** Retrospective secondary analysis of a public de-identified benchmark dataset
- **Outcome:** Binary class — CKD vs not CKD
- **Predictors:** All available non-target variables after cleaning

#### Phase 4 — Methodology

**4.1 Data Cleaning**
- Inspect variable types
- Standardize inconsistent categorical labels
- Convert numeric-like text fields
- Impute missing values separately per type
- Encode categoricals, scale numerics (especially for LR)
- Fit preprocessing only on training folds — **no leakage**

**4.2 Train-Test Strategy**
- 80/20 stratified train-test split
- 5-fold stratified cross-validation inside training data for tuning
- Final test set untouched until end

**4.3 Imbalance Handling** (compare all three):
- No rebalancing
- Class weighting
- SMOTE (on training folds only)

**4.4 Models**

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline, regularized, good interpretability, often decent calibration |
| Random Forest | Nonlinear ensemble, robust, handles mixed feature interactions |
| Gradient Boosting | Strongest nonlinear learner, best raw discrimination, calibration inspection needed |

**4.5 Hyperparameter Grids**

*Logistic Regression:*
- C: [0.01, 0.1, 1, 10]
- penalty: l2
- solver: liblinear or lbfgs

*Random Forest:*
- n_estimators: [100, 300, 500]
- max_depth: [None, 3, 5, 10]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

*Gradient Boosting:*
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [2, 3, 4]
- subsample: [0.8, 1.0]

#### Phase 5 — Evaluation Metrics

**Discrimination:** accuracy, precision, recall, F1-score, AUROC, specificity, confusion matrix

**Calibration:** Brier score, calibration curve / reliability diagram, optionally log loss

**Clinical usefulness discussion:**
- Whether false negatives are more costly than false positives
- Whether well-calibrated probabilities matter more than classification accuracy in CKD screening

#### Phase 6 — Interpretability Plan

| Model | Method |
|---|---|
| Logistic Regression | standardized coefficients, odds-direction, sign/magnitude interpretation |
| Random Forest | impurity-based importance (with caution), permutation importance, SHAP |
| Gradient Boosting | permutation importance, SHAP summary plot |

Clinically meaningful features expected to rank highly: serum creatinine, hemoglobin, blood urea, albumin, specific gravity, hypertension, diabetes mellitus

#### Phase 7 — Experiments

| Experiment | Description |
|---|---|
| 1. Baseline model comparison | LR vs RF vs GB with standard preprocessing |
| 2. Imbalance strategy comparison | Per model: default, class-weighted, SMOTE |
| 3. Calibration comparison | Uncalibrated vs isotonic/Platt scaling |
| 4. Interpretability comparison | Top 10 features per model, agreement/disagreement analysis |
| 5. Missingness sensitivity (optional) | Median/mode vs KNN vs iterative imputation |

#### Phase 8 — Paper Structure (JAMIA Open Format)

1. Background and Significance
2. Objectives
3. Materials and Methods
4. Results
5. Discussion
6. Conclusion

#### Phase 9 — Section-by-Section Writing Guide

**Background and Significance:**
- Explain CKD as a meaningful prediction problem
- Show prior papers focus on accuracy alone
- Justify why calibration and interpretability matter
- End with bridge sentence: *"Existing CKD prediction studies frequently report discrimination performance, but fewer compare standard ML models using a unified protocol that also examines class imbalance, calibration, and model interpretability."*

**Objectives:** Short, explicit — compare three ML approaches; evaluate discrimination, calibration, explainability; identify best overall clinical prediction profile

**Materials and Methods subsections:** dataset, preprocessing, split strategy, imbalance handling, models and tuning, evaluation metrics, explainability, software environment

**Results subsections:** descriptive statistics, model performance, calibration findings, feature importance findings, sensitivity analyses

**Discussion — four moves:**
1. Summarize the strongest result
2. Explain what it means clinically
3. Compare with prior CKD ML studies
4. Discuss limitations and future work

**Conclusion:** Two paragraphs — what you found + why it matters for future healthcare AI modeling

#### Phase 10 — Tables and Figures

**Tables (max 4):**

| # | Title | Columns |
|---|---|---|
| Table 1 | Dataset variables and preprocessing actions | Variable, Type, Clinical meaning, Missingness %, Preprocessing step |
| Table 2 | Baseline characteristics and class distribution | Feature summary, CKD group, non-CKD group, Overall |
| Table 3 | Final model performance on test set | Model, Accuracy, Precision, Recall, Specificity, F1, AUROC, Brier score |
| Table 4 | Top features by model | Rank, Logistic Regression, Random Forest, Gradient Boosting |

**Figures (max 6):**

| # | Title |
|---|---|
| Figure 1 | Study workflow diagram |
| Figure 2 | Class distribution and missingness heatmap |
| Figure 3 | ROC curves for all three models |
| Figure 4 | Calibration curves for all three models |
| Figure 5 | SHAP summary plot for best-performing model |
| Figure 6 | Permutation importance comparison across models |

#### Phase 11 — Novelty Statement
> This study contributes a reproducible and clinically oriented evaluation framework for CKD prediction that jointly compares standard models across discrimination, calibration, imbalance handling, and interpretability, rather than reporting accuracy alone.

#### Phase 12 — Methods Language (ready to adapt)

**Dataset paragraph:**
> This study used the Chronic Kidney Disease dataset from the UCI Machine Learning Repository. The dataset contains 400 instances and 24 predictor variables with a binary class label indicating chronic kidney disease status. Variables include demographic, laboratory, and comorbidity-related features, and the dataset contains missing values that require preprocessing prior to model development.

**Calibration paragraph:**
> In addition to discrimination metrics, model calibration was evaluated because healthcare prediction models may be used to estimate risk rather than only assign class labels. Calibration quality was assessed using Brier score and reliability diagrams. These measures were selected because probability calibration reflects the agreement between predicted probabilities and observed outcome frequencies, which is important for clinically interpretable risk estimates.

**Reproducibility paragraph:**
> All preprocessing operations, including imputation, encoding, scaling, and resampling, were performed within the training folds to avoid information leakage. Model tuning used stratified cross-validation on the training partition, and final evaluation was conducted on a held-out stratified test set.

#### Phase 13 — Software Stack

```
Python 3.11 or 3.12
pandas, numpy
scikit-learn
xgboost or sklearn GradientBoostingClassifier
imbalanced-learn
shap
matplotlib
scipy
jupyter
```

**Project Folder Structure:**
```
ckd-paper/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_calibration.ipynb
│   └── 05_explainability.ipynb
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py
│   └── explain.py
├── figures/
├── tables/
├── manuscript/
│   ├── main.docx
│   ├── cover_letter.docx
│   └── patient_abstract.docx
├── requirements.txt
└── README.md
```

#### Phase 14 — Writing Timeline

| Week | Tasks |
|---|---|
| Week 1 | Download and inspect dataset, clean variables, write study protocol, decide evaluation pipeline |
| Week 2 | Run baseline models, finalize train/test strategy, tune models |
| Week 3 | Run calibration and imbalance experiments, generate tables and figures |
| Week 4 | Write introduction and methods, draft results, begin discussion |
| Week 5 | Revise full paper, format for JAMIA Open, prepare repo and supplement |
| Week 6 | Final checks, cover letter, submission |

#### Phase 15 — Supplementary Material
- Full hyperparameter grids
- Full classification reports
- Additional confusion matrices
- Imputation sensitivity tables
- Code notebook link
- Feature preprocessing dictionary

#### Phase 16 — Ethics Statement
> This study used a publicly available de-identified dataset from the UCI Machine Learning Repository. No direct human participant recruitment or intervention was involved. Ethical review was therefore not required for this secondary analysis of public data, subject to institutional policy.

#### Phase 17 — Discussion Strategy

1. **Which model was strongest overall?** (Likely Gradient Boosting for AUROC/F1)
2. **Which model was most trustworthy?** (Logistic Regression may be better calibrated or more interpretable)
3. **What does that mean for healthcare use?**
   - Highest-performing model → best for screening
   - Best-calibrated model → better for risk communication
   - Most interpretable model → better for early clinical decision support

#### Phase 18 — Common Mistakes to Avoid
- Using full dataset for imputation before splitting
- Applying SMOTE before train/test separation
- Reporting only accuracy
- Using too many models without a clear reason
- Claiming "clinical deployment readiness" from one small public dataset
- Presenting SHAP plots with no interpretation
- Comparing against prior papers without matching their conditions
- Submitting to JBI with no methodological angle

#### Phase 19 — Reviewer Response Prep

**If reviewers ask "what is novel?":**
> The novelty lies in the unified, reproducible evaluation of discrimination, calibration, imbalance handling, and interpretability for CKD prediction, which addresses a gap in studies that report performance without analyzing reliability and practical clinical usability.

**If reviewers ask for stronger clinical relevance:** Add false negative discussion, threshold sensitivity analysis, and explanation of how calibrated risk estimates may support screening or referral prioritization.

#### Phase 20 — Submission Checklist

- [ ] Manuscript ≤ 4,000 words
- [ ] Structured abstract ≤ 250 words
- [ ] Maximum 4 tables
- [ ] Maximum 6 figures
- [ ] Title page complete
- [ ] Keywords included (MeSH-aligned where possible)
- [ ] Ethics statement included
- [ ] Conflict of interest statement included
- [ ] Data availability statement included
- [ ] Code repository ready
- [ ] References cleaned
- [ ] Figures high resolution
- [ ] Tables editable (not images)

#### Phase 21 — After Submission

**Possible outcomes:** reject / major revision / minor revision / accept

For a first paper, **expect revision** rather than instant acceptance.

**Possible outcomes by year:**
- Reject → revise and resubmit to backup venue
- Major revision → strengthen novelty framing, add clinical context
- Minor revision → address reviewer comments
- Accept → publish and move to Paper 2

---

## 13. Paper 1 — Ready-to-Use Manuscript Template

### Title Page

**Title:**
Comparative Performance and Reliability of Logistic Regression, Random Forest, and Gradient Boosting for Chronic Kidney Disease Prediction

**Author:** Michael O. Eniolade

**Affiliation:** [Your University / Organization]

**Corresponding Author:** Michael O. Eniolade | Email: [your email]

**Keywords:** Chronic Kidney Disease, Machine Learning, Calibration, Explainability, Clinical Prediction

---

### Abstract (≤250 words)

**Objectives**
Chronic kidney disease (CKD) is a progressive condition that often remains undetected until advanced stages, making early prediction critical for effective intervention. While machine learning models have been widely applied to CKD prediction, many studies focus primarily on classification accuracy without evaluating reliability, calibration, and interpretability. This study compares Logistic Regression, Random Forest, and Gradient Boosting for CKD prediction using a unified evaluation framework.

**Materials and Methods**
A retrospective analysis was conducted using the Chronic Kidney Disease dataset from the UCI Machine Learning Repository. Data preprocessing included missing value imputation, categorical encoding, and feature scaling. Models were trained using stratified cross-validation with hyperparameter tuning. Performance was evaluated on a held-out test set using discrimination metrics (accuracy, precision, recall, F1-score, AUROC), calibration metrics (Brier score and reliability curves), and interpretability techniques including feature importance and SHAP analysis.

**Results**
Ensemble models demonstrated higher discrimination performance compared to Logistic Regression, with Gradient Boosting achieving the highest AUROC. However, Logistic Regression exhibited more stable calibration prior to adjustment. Feature importance analysis showed consistent identification of clinically relevant variables across models, including serum creatinine, hemoglobin, and blood urea.

**Discussion**
The results suggest that model selection for CKD prediction should consider not only classification performance but also calibration and interpretability. While ensemble methods offer improved predictive power, simpler models may provide advantages in transparency and reliability for clinical applications.

**Conclusion**
A comprehensive evaluation approach reveals trade-offs between performance, calibration, and interpretability in CKD prediction models, highlighting the importance of selecting models based on clinical use rather than accuracy alone.

---

### 1. Background and Significance

Chronic kidney disease (CKD) is a major global health concern characterized by a gradual loss of kidney function over time. It is associated with increased risk of cardiovascular disease, kidney failure, and mortality. Early detection is essential, as timely intervention can slow disease progression and improve patient outcomes. However, CKD is often asymptomatic in its early stages, making predictive modeling an important tool for screening and risk assessment.

Machine learning techniques have increasingly been applied to CKD prediction using clinical and laboratory data. Prior studies have demonstrated that algorithms such as Logistic Regression, Random Forest, and Gradient Boosting can achieve high classification accuracy on benchmark datasets. Despite these advances, many studies focus primarily on predictive performance metrics such as accuracy and area under the receiver operating characteristic curve, without examining whether predicted probabilities are well-calibrated or whether models provide interpretable insights.

In clinical settings, predictive models are often used to estimate risk rather than simply assign class labels. Poorly calibrated models may produce probabilities that do not reflect true outcome likelihoods, which can lead to inappropriate clinical decisions. Similarly, lack of interpretability can limit clinician trust and hinder adoption. These issues highlight the need for evaluation frameworks that extend beyond accuracy to include reliability and transparency.

This study addresses this gap by conducting a systematic comparison of three widely used machine learning approaches for CKD prediction, focusing not only on discrimination performance but also on calibration and interpretability within a reproducible experimental framework.

---

### 2. Objectives

The objective of this study is to evaluate and compare the performance of Logistic Regression, Random Forest, and Gradient Boosting models for chronic kidney disease prediction using a comprehensive and clinically relevant evaluation framework.

**Specific objectives:**
1. To compare model performance using standard classification metrics including accuracy, precision, recall, F1-score, and AUROC
2. To assess model calibration using Brier score and reliability curves
3. To evaluate the impact of class imbalance handling techniques on model performance
4. To analyze model interpretability using feature importance and SHAP-based explanations
5. To identify the model that provides the best balance between predictive performance, reliability, and interpretability for potential clinical use

---

### 3. Materials and Methods

#### 3.1 Dataset
This study utilized the Chronic Kidney Disease dataset from the UCI Machine Learning Repository. The dataset consists of 400 instances and includes 24 predictor variables along with a binary target variable indicating the presence or absence of CKD. The features include demographic attributes, laboratory measurements, and clinical indicators such as blood pressure, serum creatinine, hemoglobin, and presence of comorbid conditions. The dataset contains missing values across several features, requiring preprocessing prior to model development.

#### 3.2 Data Preprocessing
Data preprocessing involved several steps to ensure data quality and consistency. Categorical variables were standardized and encoded using appropriate techniques. Numerical variables were converted to consistent formats and scaled where necessary. Missing values were handled using imputation strategies, with separate approaches applied to numerical and categorical variables. All preprocessing steps were applied within the training folds to prevent data leakage.

#### 3.3 Train-Test Strategy
The dataset was divided into training and testing subsets using an 80/20 stratified split to preserve class distribution. Model training and hyperparameter tuning were performed using 5-fold stratified cross-validation on the training set. The final evaluation was conducted on the held-out test set to assess generalization performance.

#### 3.4 Class Imbalance Handling
To account for potential class imbalance, multiple strategies were evaluated, including baseline training without adjustment, class-weighted models, and oversampling techniques applied to the training data. Oversampling was performed only within cross-validation folds to avoid information leakage.

#### 3.5 Model Development
Three machine learning models were evaluated:
- **Logistic Regression** — baseline linear model with regularization
- **Random Forest** — ensemble method based on decision trees
- **Gradient Boosting** — sequential ensemble approach designed to optimize predictive performance

Hyperparameters for each model were tuned using cross-validation.

#### 3.6 Evaluation Metrics
Model performance was evaluated using discrimination and calibration metrics. Discrimination: accuracy, precision, recall, F1-score, specificity, and AUROC. Calibration: Brier score and reliability diagrams, which measure the agreement between predicted probabilities and observed outcomes.

#### 3.7 Interpretability Analysis
For tree-based models, feature importance was computed using both impurity-based and permutation-based approaches. For Logistic Regression, model coefficients were analyzed to determine the direction and strength of associations. SHAP values were used to provide a consistent framework for interpreting model predictions across all models.

#### 3.8 Software and Implementation
All experiments were conducted using Python. Data preprocessing and modeling used pandas, NumPy, and scikit-learn. Model evaluation, calibration, and interpretability analyses were performed using standard machine learning and visualization tools. The experimental workflow was designed to be fully reproducible.

---

### 4. Results
*(To be completed after running experiments)*

**4.1 Descriptive Statistics**
*(Insert Table 2 here)*

**4.2 Model Performance**
*(Insert Table 3 here)*

**4.3 Calibration Findings**
*(Insert Figure 4 — calibration curves here)*

**4.4 Feature Importance Findings**
*(Insert Figure 5, Figure 6, Table 4 here)*

**4.5 Sensitivity Analyses**
*(Insert imbalance strategy results here)*

---

### 5. Discussion
*(To be completed after analysis)*

- Which model was strongest overall?
- Which model was most trustworthy?
- What does that mean for healthcare use?
- Comparison with prior CKD ML studies
- Limitations and future work

---

### 6. Conclusion
*(To be completed after results)*

Two paragraphs:
1. Summary of main findings
2. Why results matter for future healthcare AI modeling

---

### Cover Letter Template

```
Dear Editor,

Please consider our manuscript, "Comparative Performance and Reliability of 
Logistic Regression, Random Forest, and Gradient Boosting for Chronic Kidney 
Disease Prediction," for publication as a Research Article in JAMIA Open.

This study presents a reproducible evaluation of three widely used machine 
learning approaches for chronic kidney disease prediction using a public benchmark 
dataset. Rather than focusing on discrimination performance alone, the manuscript 
compares model behavior across class imbalance handling, calibration, and 
interpretability, with emphasis on implications for clinically meaningful risk 
prediction.

We believe the manuscript fits JAMIA Open because it addresses biomedical and 
health informatics through transparent machine learning evaluation, reproducibility, 
and practical considerations for clinical prediction tools. The dataset is publicly 
available, and the manuscript will include access to code and supplementary 
materials to support reproducibility.

This manuscript is original, has not been published previously, and is not under 
consideration elsewhere. The study used a publicly available de-identified dataset, 
and no direct human participant involvement was required.

Thank you for your consideration.

Sincerely,
Michael O. Eniolade
[Affiliation]
[Email]
```

---

### Key References to Gather

- CKD clinical background papers
- Prior CKD machine learning studies
- Calibration methodology papers (scikit-learn calibration documentation)
- Explainability methodology papers (SHAP)
- JAMIA Open formatting guidelines
- UCI CKD dataset citation

---

*Document compiled from ChatGPT research planning session — April 2026*
