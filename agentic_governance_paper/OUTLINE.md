# Paper 2 — Strategic Outline
## Governing Trustworthy Agentic Healthcare AI: A Calibration-First Framework with Distributed Ledger Accountability

**Author:** Michael O. Eniolade  
**Institution:** University of the Cumberlands  
**Target Venue:** Blockchain in Healthcare Today (BHTY) — Special Issue: Ethics of AI Agents and Agentic AI in Healthcare  
**Paper Type:** Perspective / Conceptual Framework  
**Submission Deadline:** June 30, 2026 (December 2026 Issue)  
**Backup Option:** BHTY General CFP — August 2026 Issue (rolling)  
**Target Word Count:** 3,500–5,000 words  
**APC Note:** Student APC = $850 USD. Request waiver if available.

---

## One-Sentence Contribution

This paper proposes a five-layer trustworthiness governance framework for agentic healthcare AI that extends existing model validation standards — specifically calibration integrity, uncertainty quantification, fairness auditing, and deployment readiness scoring — with blockchain-based distributed ledger infrastructure for immutable accountability, verifiable provenance, and patient-controlled consent management.

---

## The Core Argument (5 Bullets)

1. **The agentic shift changes the governance problem.** Healthcare AI is transitioning from single-inference tools to autonomous agents that make sequential, compounding decisions. Existing governance frameworks (TRIPOD+AI, FDA AI/ML guidance, EU AI Act) were designed for static models and do not address pipeline-level reliability, inter-step accountability, or emergent behavior in agentic systems.

2. **Trustworthiness requires more than accuracy at deployment.** Current evaluation focuses on pre-deployment discrimination metrics (AUROC, accuracy). Agentic AI also demands: calibrated probability outputs at each decision step, distribution-free uncertainty guarantees (conformal prediction), subgroup fairness auditing, and scored deployment readiness — criteria that existing governance frameworks leave unmeasured.

3. **When agents cause harm, we often cannot reconstruct why.** The accountability gap in agentic AI is epistemic, not just legal. Multi-step autonomous decisions leave no durable, verifiable audit record. Traditional logging is mutable, siloed, and institution-specific. This makes post-incident investigation and regulatory review unreliable.

4. **Blockchain/DLT provides the accountability infrastructure layer, not the trustworthiness solution.** Distributed ledger technologies offer: immutable audit trails of agent decisions and reasoning steps; smart contracts encoding ethical and operational constraints as executable rules; decentralized patient consent management that persists across AI actions; and verifiable model provenance (training data lineage, version history, deployment records). DLT makes trustworthiness evidence *verifiable* — it does not substitute for producing that evidence.

5. **The framework is actionable and connects to emerging clinical AI standards.** Each layer maps to measurable criteria — calibration thresholds (ECE ≤ 0.10), coverage guarantees (conformal ≥ 0.90), subgroup equity bounds, deployment readiness scoring — providing a concrete protocol rather than abstract principles. The framework is designed to be validated empirically in subsequent research.

---

## Proposed Sections

| # | Section | Target Words |
|---|---------|-------------|
| 1 | Introduction — The agentic shift and the governance gap | ~500 |
| 2 | What makes agentic healthcare AI different | ~400 |
| 3 | The trustworthiness gap in current governance frameworks | ~500 |
| 4 | A five-layer trustworthiness governance framework | ~700 |
| 5 | Distributed ledger technologies as accountability infrastructure | ~600 |
| 6 | Implementation challenges | ~400 |
| 7 | Research agenda | ~400 |
| 8 | Conclusion | ~200 |
| — | References | ~20 citations |

**Total:** ~3,700 words body + abstract + references

---

## Five-Layer Framework Summary

| Layer | What It Governs | Key Metric/Mechanism |
|-------|----------------|----------------------|
| 1. Reliability Evaluation | Calibration quality, uncertainty at each agent decision | ECE ≤ 0.10, conformal coverage ≥ 0.90 |
| 2. Fairness Auditing | Subgroup calibration equity, error disparity | Max subgroup ECE gap ≤ 0.05, FNR parity |
| 3. Provenance Verification | Model lineage, training data, version history | On-chain model registry, data hashes |
| 4. Consent Management | Patient-controlled scope of autonomous AI action | Smart contracts encoding consent boundaries |
| 5. Deployment Readiness | Pre-deployment and ongoing readiness scoring | 8-criterion checklist, ongoing drift monitoring |

---

## Connection to Dissertation Roadmap

| Dissertation Paper | How This Paper Relates |
|-------------------|----------------------|
| Paper 1 (CKD Calibration) | Layers 1 and 5 are the technical implementation of what this paper argues for theoretically |
| Paper 3 (Secure AI Architecture) | Layer 3–4 (provenance, consent) maps to the secure architecture design |
| Paper 6 (Reliability Framework) | Layer 1–2 are the empirical operationalization of this conceptual framework |
| Paper 9 (Flagship CDSS Framework) | This paper is the governance theory that Paper 9 implements |
| Paper 10 (Deployment Readiness) | Layer 5 is the precursor to the full deployment readiness protocol |

This paper functions as the **theoretical and governance foundation** for the empirical papers that follow. It can be cited in every subsequent paper in the roadmap.

---

## Key References Needed

- Guo et al. 2017 — Calibration of modern neural networks (already in first_paper/references)
- Angelopoulos & Bates 2022 — Conformal prediction (already in first_paper/references)
- Collins et al. 2024 — TRIPOD+AI (already in first_paper/references)
- Vasey et al. 2022 — DECIDE-AI (already in first_paper/references)
- Campagner et al. 2025 — Uncertainty-aware ML in healthcare (already in first_paper/references)
- FDA guidance on AI/ML-based Software as Medical Device (SaMD)
- EU AI Act provisions for high-risk AI systems (Title III)
- Recent agentic AI in healthcare literature (2024–2025)
- Blockchain audit trail papers in healthcare (2023–2025)
- Smart contract healthcare governance literature

---

## Submission Checklist

- [ ] Abstract ≤ 300 words (unstructured — perspectives category)
- [ ] Cover letter stating: "Ethics of AI Agents and Agentic AI in Healthcare" as Theme Issue title
- [ ] Manuscript as Word document per BHTY author guidelines
- [ ] Vancouver reference style (numbered)
- [ ] ≤ 10,000 words including references
- [ ] No IRB required (conceptual/perspective paper — no human subjects data)
- [ ] Submit via: https://blockchainhealthcaretoday.com/index.php/journal/about/submissions
