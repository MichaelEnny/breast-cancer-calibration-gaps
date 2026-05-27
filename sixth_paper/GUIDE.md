# Paper 6 (Supplementary) — Step-by-Step Execution Guide
### Teaching Trustworthy Healthcare AI: Preparing Learners to Evaluate Reliability, Fairness, Security, and Deployment Readiness

**Author:** Michael O. Eniolade — University of the Cumberlands  
**Type:** Book chapter / Encyclopedia entry  
**Target Publication:** Encyclopedia of Education and Human Development  
**Role in Dissertation:** Supplementary contribution — translates the technical framework from Papers 1, 3, 6, 8, 9, and 10 into an education-facing teaching resource  
**Status:** Proposal written (see `proposal.txt`)

---

## Important: What This Contribution Is and Is Not

This is **not** a research paper. It is an **invited or submitted book chapter** for an educational encyclopedia. Unlike the 10 research papers, it does not require:
- A dataset
- New metrics
- Experimental results
- Statistical analysis

What it requires:
- A clear, structured pedagogical framework
- Realistic classroom case studies
- Ready-to-use teaching materials (learning objectives, activities, assessments)
- Clear connection between technical content from the dissertation and teachable concepts for non-technical or semi-technical audiences

The chapter's impact comes from **reach**: healthcare educators, nursing informatics programs, public health schools, and data science courses will use this to train the next generation of clinical AI evaluators. That reach is the contribution.

---

## Table of Contents

1. [Understanding the Target Venue](#1-understanding-the-target-venue)
2. [Chapter Scope and Audience](#2-chapter-scope-and-audience)
3. [The Teaching Framework: Five Pillars](#3-the-teaching-framework-five-pillars)
4. [Chapter Outline](#4-chapter-outline)
5. [Case Study Development Guide](#5-case-study-development-guide)
6. [Learning Objectives](#6-learning-objectives)
7. [Classroom Activities](#7-classroom-activities)
8. [Assessment Ideas](#8-assessment-ideas)
9. [Section-by-Section Writing Guide](#9-section-by-section-writing-guide)
10. [Figures and Tables Plan](#10-figures-and-tables-plan)
11. [Connecting to the Research Papers](#11-connecting-to-the-research-papers)
12. [Writing Style Guide for Educational Content](#12-writing-style-guide-for-educational-content)
13. [Folder Structure](#13-folder-structure)
14. [Timeline](#14-timeline)
15. [Submission Checklist](#15-submission-checklist)

---

## 1. Understanding the Target Venue

### Encyclopedia of Education and Human Development

Encyclopedias in this category publish reference-quality chapters used by educators, curriculum developers, and graduate students. Key characteristics:

- **Audience:** Educators, curriculum developers, graduate students, healthcare professionals — NOT necessarily ML researchers
- **Tone:** Authoritative but accessible — no assumed deep ML knowledge
- **Length:** Typically 4,000–8,000 words (confirm with editor)
- **Format:** Structured with clear headings; practical materials expected; references to seminal works
- **What reviewers look for:** Pedagogical clarity, practical usability, grounding in current evidence, connection to professional practice

**Before submitting:** Contact the encyclopedia editor to confirm:
1. Whether the chapter is invited or submitted via open call
2. Exact word limit and formatting requirements
3. Required permissions for any figures or case materials
4. Whether classroom activities and assessments are expected or optional

### Alternative Venues (if encyclopedia submission does not proceed)

- *Journal of the American Medical Informatics Association* (JAMIA) — Education section
- *npj Digital Medicine* — Perspectives or Comment article
- *BMC Medical Education* — full article format
- *Academic Medicine* — innovations in medical education format

---

## 2. Chapter Scope and Audience

### Primary Audiences

| Audience | What they need from this chapter |
|---|---|
| Health informatics educators | A teachable framework with ready-to-use classroom materials |
| Graduate students (data science, health informatics, public health) | A practical vocabulary for evaluating clinical AI systems |
| Healthcare professionals (clinicians, administrators) | A non-technical way to ask the right questions before using or procuring an AI system |
| Nursing informatics instructors | Cases and examples grounded in clinical workflow relevance |

### What the Chapter Does NOT Assume

- Knowledge of Python, scikit-learn, or any specific ML library
- Ability to compute calibration metrics independently
- Prior coursework in statistics beyond basic probability

### What the Chapter DOES Assume

- Basic familiarity with what machine learning means (classification, prediction)
- A healthcare or health informatics professional context
- Interest in responsible AI adoption, not just AI capability

---

## 3. The Teaching Framework: Five Pillars

The chapter organizes trustworthy healthcare AI evaluation around five pillars. These map directly to the dissertation's research contributions:

| Pillar | Core Question | Dissertation Paper |
|---|---|---|
| **Reliability** | Are the model's probability outputs accurate? Does it degrade over time? | Papers 1, 6, 7 |
| **Fairness** | Does the model perform equally well and communicate risk equally accurately for all patient groups? | Papers 4, 8 |
| **Security** | Is patient data protected? Can the system be audited? Is it resilient to manipulation? | Paper 3 |
| **Explainability** | Can clinicians understand why the model gave a specific output? | Papers 6, 9 |
| **Deployment Readiness** | Has this system been evaluated end-to-end before reaching patients? | Papers 10, 9 |

### The Teaching Sequence

Introduce the pillars in this order — each builds on the previous:
1. Start with Reliability (connects to familiar concept: accuracy is not enough)
2. Add Fairness (who is accuracy not enough for?)
3. Add Security (what happens when the system is attacked or misused?)
4. Add Explainability (can we trust what we can't explain?)
5. Synthesize with Deployment Readiness (all five together before patients are affected)

---

## 4. Chapter Outline

### Recommended Structure (8 sections + references + teaching appendix)

```
1. Introduction: Why Accuracy Is Not Enough
2. Pillar 1: Reliability — Calibration, Uncertainty, and Drift
3. Pillar 2: Fairness — Subgroup Performance and Calibration Equity
4. Pillar 3: Security — Data Protection, Audit, and Adversarial Risk
5. Pillar 4: Explainability — What Clinicians Need to Understand
6. Pillar 5: Deployment Readiness — From Validation to Practice
7. Case Studies (at least two)
8. Implications for Healthcare AI Education
References
Teaching Appendix (learning objectives, activities, assessments)
```

---

## 5. Case Study Development Guide

Two case studies are required. They should be realistic, clinically grounded, and teach a specific concept from each pillar.

---

### Case Study 1: The Calibration Trap — Two CKD Models

**Clinical context:** A hospital is choosing between two AI tools for chronic kidney disease screening in a primary care clinic.

**Setup:**
- Model A: Random Forest with AUROC = 0.98, ECE = 0.34
- Model B: Logistic Regression with AUROC = 0.91, ECE = 0.05

Model A has dramatically better discrimination. Model B has dramatically better calibration.

**The teaching moment:** Model A tells you WHO is likely to have CKD very accurately — but its probability estimates are wrong. If the clinic uses predicted probabilities to decide how urgently to refer a patient (a 40% risk patient gets a faster referral than a 15% risk patient), then Model A's miscalibrated probabilities could lead to inappropriate urgency decisions even though its classification is excellent.

**Discussion questions:**
1. Which model would you choose for a clinic that only wants to identify high-risk patients for a binary screening decision?
2. Which model would you choose for a clinic that uses the predicted probability to prioritize follow-up appointments?
3. What additional information would you want before choosing?

**Connection to Pillar 1 (Reliability):** Introduces ECE, reliability diagrams, and the distinction between discrimination and calibration in a clinically intuitive way.

---

### Case Study 2: The Invisible Gap — An ED Triage Model

**Clinical context:** A hospital deploys an AI triage tool to help nurses prioritize patients in the emergency department. The tool is validated on 50,000 ED visits and shows overall AUROC = 0.87 and ECE = 0.06.

**The hidden problem:** When performance is broken down by demographic subgroup:
- White patients aged 40–60: AUROC = 0.89, ECE = 0.04
- Black patients aged 75+: AUROC = 0.71, ECE = 0.29

**The teaching moment:** The model passed all standard validation metrics. The aggregate numbers looked good. The subgroup analysis reveals that elderly Black patients receive substantially less reliable risk estimates — both in terms of ranking accuracy and probability accuracy. This subgroup represents a higher proportion of high-acuity visits than their sample proportion suggests.

**Discussion questions:**
1. Should this model have been deployed?
2. What additional validation should have been required before deployment?
3. If the hospital discovers this gap six months after deployment, what should they do?
4. Who is responsible for identifying this gap — the developers, the hospital IT team, the clinical leadership, or someone else?

**Connection to Pillars 2 and 5 (Fairness and Deployment Readiness):** Teaches subgroup analysis, calibration equity, and why deployment readiness requires more than aggregate validation.

---

### Case Study 3 (Optional — Security): The Audit Trail Question

**Clinical context:** A radiology department uses an AI tool that flags chest X-rays for pneumonia. A patient files a complaint that their scan was misread. The hospital's IT team is asked to reconstruct what the AI predicted and why.

**The hidden problem:** The AI system has no audit trail. The prediction is not logged. The SHAP explanation was never recorded. It is impossible to reconstruct what the AI said or why.

**Discussion questions:**
1. What security and governance controls should have been in place?
2. Who bears responsibility for the missing audit trail?
3. What would a deployment-ready system have provided that would make this reconstruction possible?

**Connection to Pillars 3 and 5 (Security and Deployment Readiness).**

---

## 6. Learning Objectives

After completing this chapter, the learner will be able to:

**Knowledge objectives:**
1. Define the five pillars of trustworthy healthcare AI: reliability, fairness, security, explainability, and deployment readiness
2. Distinguish between discrimination performance (AUROC, accuracy) and calibration quality (ECE, Brier score)
3. Explain what subgroup performance analysis means and why aggregate metrics can hide harm

**Skill objectives:**
4. Identify which evaluation questions to ask when reviewing a clinical AI system
5. Apply a structured evaluation framework (based on CADRI) to assess whether a clinical AI system is ready for deployment
6. Recognize red flags in a clinical AI validation report that suggest a system may not be trustworthy

**Disposition objectives:**
7. Develop professional responsibility for asking evaluation questions about AI systems before adoption
8. Recognize that trustworthy AI requires both technical rigor and governance accountability

---

## 7. Classroom Activities

### Activity 1: Model Comparison (45 minutes)

**Level:** Graduate health informatics, nursing informatics, data science  
**Materials needed:** Two printed/shared "model validation reports" (create fictional but realistic one-page summaries)

**Instructions:**
- Students receive two model validation reports for competing CKD prediction tools
- Report A shows high AUROC, high accuracy, no subgroup analysis, no calibration metrics
- Report B shows lower AUROC, calibration metrics reported, subgroup analysis included, deployment checklist completed
- In small groups (3–4 students), answer: Which model would you recommend for clinical use? Why?
- Class debrief: compare group recommendations; identify what information each group used and missed

**Learning outcome:** Students learn to identify absent information in validation reports as a red flag, not just evaluate information that is present.

---

### Activity 2: The Deployment Readiness Audit (60 minutes)

**Level:** Graduate or professional development  
**Materials needed:** Simplified CADRI-based checklist (10 items); a fictional clinical AI case description (2 pages)

**Instructions:**
- Students receive a fictional case: a hospital considering deploying an AI sepsis early warning system
- The case includes information about the system's development but has deliberate gaps (no external validation, no subgroup analysis, no audit trail)
- Students apply the 10-item deployment readiness checklist and assign a score
- Students write a one-paragraph recommendation: deploy, deploy with conditions, or do not deploy
- Class comparison: how did different students weight different missing items?

**Learning outcome:** Students practice structured evaluation thinking and learn to make deployment recommendations under uncertainty.

---

### Activity 3: Who Is Harmed? (30 minutes — discussion)

**Level:** Any healthcare professional audience  
**Materials needed:** The ED triage case study (Case Study 2 from Section 5)

**Instructions:**
- Present the case study
- Discussion questions: (1) Who is harmed by this system? (2) When does the harm occur? (3) Who should have caught this? (4) What should happen next?
- No right answer — the goal is developing professional vocabulary for discussing AI harm

**Learning outcome:** Builds capacity to discuss AI harm in clinical settings without requiring technical expertise.

---

## 8. Assessment Ideas

### Assessment 1: Clinical AI Validation Report Review (Individual)

**Task:** Provide students with a real (published) clinical AI validation study. Ask them to evaluate it against the five pillars using a structured rubric:
- Does the paper report calibration metrics? (Yes/No + what is missing)
- Does the paper perform subgroup analysis? (Yes/No + which groups)
- Does the paper address security or governance? (Yes/No)
- Does the paper address explainability? (Yes/No)
- Would this system pass a deployment readiness review? (Justify in 200 words)

**Learning outcome:** Students apply the framework to real published work and develop critical reading skills for clinical AI literature.

---

### Assessment 2: Deployment Readiness Recommendation (Group)

**Task:** Groups of 3–4 are given a clinical AI system description (fictional but realistic, 3–4 pages). They must:
1. Complete a CADRI-style scoring rubric
2. Write a deployment recommendation memo (1 page, non-technical language) addressed to a hospital clinical leadership team
3. Present key findings in a 5-minute presentation

**Learning outcome:** Students translate technical evaluation into professional communication — a critical skill for health informatics professionals.

---

### Assessment 3: Short Essay — "What Should Clinicians Ask?"

**Task:** In 500 words, a student writes: "You are a clinician who has been told that your hospital is implementing an AI triage tool next month. What three questions do you ask before it goes live, and why?"

**Learning outcome:** Develops the disposition that clinicians — not just developers — have responsibility to ask about AI trustworthiness.

---

## 9. Section-by-Section Writing Guide

### Section 1 — Introduction: Why Accuracy Is Not Enough (approximately 600 words)

**Opening hook:** Start with a concrete scenario — not a statistic. Example: "Imagine two AI tools for predicting kidney disease risk. Both achieve 95% accuracy on their test set. But one model is wrong about risk level 30% of the time for elderly Black patients, and its probability estimates are systematically too low for patients who are later admitted to hospital. The other is slightly less accurate but equally reliable for every patient group. Which tool would you trust with your patients?"

**Key points to establish:**
- Accuracy tells you the average — it does not tell you about variability, reliability, or fairness
- Healthcare AI is not used to produce predictions in a vacuum — it communicates risk to clinicians and patients who act on it
- The five pillars emerge from a single question: what does it mean for a healthcare AI system to be trustworthy, not just accurate?

**Do not:** Start with a definition of machine learning. The audience knows what AI is. Start with the clinical relevance.

---

### Section 2 — Pillar 1: Reliability (approximately 800 words)

**Teach:** Calibration, ECE (explain as "the gap between what the model claims and what actually happens"), Brier score (explain as "the average squared error of probability predictions — lower is better"), uncertainty communication, model drift.

**Use:** Case Study 1 (CKD) to illustrate calibration vs. discrimination.

**Key teaching point:** Probability estimates are clinical communications. A model that says "40% risk" is telling a clinician and patient something. If that number is systematically wrong for certain patients, it is a communication failure with clinical consequences.

**Avoid:** Mathematical formulas. Use plain language descriptions and visual analogies (reliability diagram described as: "imagine plotting the model's confidence against how often it was right — a trustworthy model's dots should fall on a straight diagonal line").

---

### Section 3 — Pillar 2: Fairness (approximately 700 words)

**Teach:** Subgroup performance analysis, the difference between classification fairness (equal AUROC) and calibration equity (equal probability reliability), the concept of intersectional fairness (performance can look acceptable for each attribute separately while hiding a gap for specific combinations).

**Use:** Case Study 2 (ED triage) to illustrate how aggregate metrics hide subgroup harm.

**Key teaching point:** The question is not just "does the model work?" but "does it work equally well for all the patients it will be used on?" These are different questions with different answers.

---

### Section 4 — Pillar 3: Security (approximately 500 words)

**Teach:** RBAC, audit trails, adversarial inputs (explain as: inputs designed to fool the model into producing wrong outputs), privacy protections.

**Use:** Case Study 3 (audit trail scenario) to illustrate why governance matters.

**Key teaching point:** Healthcare AI security is not just about preventing hackers. It is about ensuring that every prediction can be explained, traced, and reviewed — which is a professional accountability requirement, not just a technical one.

---

### Section 5 — Pillar 4: Explainability (approximately 500 words)

**Teach:** SHAP values (explained as: "a way of measuring how much each feature pushed the model's prediction up or down for a specific patient"), local vs. global explanation, the difference between a model that is technically explainable and one whose explanations are clinically usable.

**Key teaching point:** An explanation that requires a data scientist to interpret is not a clinician-facing explanation. Usability is part of explainability.

---

### Section 6 — Pillar 5: Deployment Readiness (approximately 600 words)

**Teach:** CADRI as a concept (not the technical formula — the idea that deployment readiness can be scored across technical and operational dimensions), the technical/operational distinction, what governance readiness means in practice.

**Key teaching point:** Deployment readiness is not a binary gate. It is a scored profile that tells you where the gaps are. A system can be technically excellent but operationally not ready (no incident response plan, no monitoring, no clinician training materials). Both dimensions must be addressed.

---

### Section 7 — Case Studies (approximately 800 words)

Present the two required case studies (Sections 5.1 and 5.2) in the chapter body. The optional third case (security/audit) can appear in the Teaching Appendix if word count is tight.

---

### Section 8 — Implications for Healthcare AI Education (approximately 500 words)

**Key points:**
1. Healthcare AI literacy is a professional competency — not an optional add-on for technically-minded students
2. The five-pillar framework provides a shared vocabulary that clinicians, administrators, and developers can use together
3. Curriculum integration: where in health informatics, nursing informatics, and public health curricula does this content belong?
4. Future challenges: as AI systems become more complex (LLM-based clinical reasoning, multimodal AI), the evaluation vocabulary must evolve — but the five pillars provide a stable foundation

---

## 10. Figures and Tables Plan

### Required Figures

| # | Title | Description |
|---|---|---|
| Figure 1 | The Five Pillars of Trustworthy Healthcare AI | Simple visual showing the five pillars as columns supporting a "clinical deployment" roof |
| Figure 2 | Calibration Diagram (reliability plot) | Plain-language annotated version: "perfect calibration" diagonal + example of an overconfident model |
| Figure 3 | The Subgroup Analysis Gap | Bar chart showing overall AUROC = 0.87 with subgroup breakdown revealing 0.71 for the high-risk subgroup |
| Figure 4 | Deployment Readiness Profile | Spider/radar chart showing five CADRI dimensions for a fictional model |

### Required Tables

| # | Title |
|---|---|
| Table 1 | Five Pillars: definition, key questions, and evaluation methods |
| Table 2 | Simplified CADRI checklist for classroom use (10 items, scored 0–2 each) |
| Table 3 | Sample learning objectives by audience type (graduate students, clinicians, administrators) |

---

## 11. Connecting to the Research Papers

The chapter should reference the dissertation papers naturally — not as citations of your own work in a self-promotional way, but as the evidence base that motivates the teaching content.

| Chapter section | Dissertation evidence to cite |
|---|---|
| Calibration matters | Paper 1 finding: AUROC 1.00 internally, AUROC 0.48–0.58 externally — calibration collapsed completely |
| Subgroup harm is hidden by aggregates | Paper 4 finding: intersectional triage fairness gaps invisible to marginal analysis |
| Deployment readiness is measurable | Paper 10 (CADRI) — introduce CADRI as the research basis for the teaching framework |
| Reliability can be systematically evaluated | Paper 6 (CRI) — describe as the tool that underpins the reliability pillar |

Do not cite these papers as "my work." Describe the findings as empirical evidence and cite normally. Reviewers do not need to know these are your own papers.

---

## 12. Writing Style Guide for Educational Content

### Do

- Use clinical scenarios before technical definitions
- Define every technical term the first time it appears (ECE, SHAP, RBAC)
- Use analogies: "Think of a weather forecast — a good forecast says '70% chance of rain' and it actually rains 70% of the time. Model calibration works the same way."
- Write for a reader who is intelligent and professional but not a data scientist
- End each pillar section with a "Key Questions for Practice" box — 3–5 bullet questions a clinician or administrator can ask

### Do Not

- Use equations in the main text (include in a footnote or appendix if needed)
- Assume the reader has read the other papers in the dissertation
- Write in a way that requires ML knowledge to understand
- Over-cite your own work in a way that reads as self-promotion

---

## 13. Folder Structure

```
sixth_paper/
├── proposal.txt              # EXISTING — do not modify
├── GUIDE.md                  # This file
├── manuscript/
│   ├── chapter_draft.docx    # Main chapter manuscript
│   └── teaching_appendix.docx
├── case_studies/
│   ├── case1_ckd.md          # Full case study 1 write-up
│   ├── case2_triage.md       # Full case study 2 write-up
│   └── case3_audit.md        # Optional case study 3
├── teaching_materials/
│   ├── learning_objectives.md
│   ├── activity1_model_comparison.md
│   ├── activity2_deployment_audit.md
│   ├── activity3_discussion.md
│   ├── assessment_rubric.docx
│   └── simplified_cadri_checklist.pdf
├── figures/
│   ├── five_pillars_diagram.png
│   ├── calibration_diagram.png
│   ├── subgroup_gap_chart.png
│   └── deployment_readiness_radar.png
└── references/
```

---

## 14. Timeline (Estimated 12 weeks)

| Weeks | Tasks |
|---|---|
| 1 | Contact encyclopedia editor: confirm chapter is welcome, get word limit and formatting requirements |
| 2 | Confirm chapter scope with editor; finalize outline |
| 3–4 | Write case studies (all three); develop simplified CADRI checklist; write learning objectives |
| 5–6 | Write Sections 1–4 (Introduction + Pillars 1 and 2) |
| 7–8 | Write Sections 5–8 (Pillars 3–5 + Implications) |
| 9 | Create all figures; build teaching appendix (activities + assessments) |
| 10 | Full chapter review; check reading level (aim for clear professional prose, not academic jargon) |
| 11 | Final revision; format per editor's requirements |
| 12 | Submit |

---

## 15. Submission Checklist

- [ ] Editor confirmation received: chapter is accepted for submission or invited
- [ ] Word count confirmed with editor (typical: 4,000–8,000 words)
- [ ] Formatting requirements confirmed (references style, figure format, heading levels)
- [ ] Five pillars framework clearly structured with headings
- [ ] At least two case studies included with discussion questions
- [ ] Learning objectives written (knowledge, skill, and disposition levels)
- [ ] At least two classroom activities included
- [ ] At least two assessment ideas included
- [ ] Simplified CADRI checklist (10-item classroom version) included in appendix
- [ ] All figures created at publication quality
- [ ] Technical terms defined on first use
- [ ] No equations in main text
- [ ] Author affiliation, bio, and correspondence details prepared
- [ ] Permissions obtained for any non-original figures
- [ ] Conflict of interest statement included
