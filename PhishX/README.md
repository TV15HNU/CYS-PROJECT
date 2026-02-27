# PhishX: Adversarially-Aware Uncertainty-Driven Gating for QR Phishing Defense

PhishX is an advanced, multimodal deep learning framework designed to detect phishing URLs embedded in QR codes ("Quishing"). Unlike traditional detectors, PhishX utilizes **Bayesian Uncertainty Estimation** and **Adaptive Gating** to defend against sophisticated adversarial attacks that bypass static security measures.

---

## 1. Problem Statement
**The Quishing Threat:** QR Phishing (Quishing) is a rapidly growing vector where malicious URLs are obscured within QR images, bypassing traditional email gateways that only scan plaintext.
**The Generalization Gap:** Most ML models "memorize" subdomains of known phishing sites. PhishX addresses the **Domain-Level Generalization** problem by ensuring the system identifies malicious *patterns* even on entirely new, unseen domains.
**Novelty:** PhishX replaces "static" model ensembles with a risk-aware gating mechanism that shifts reliance between semantic (Transformer) and structural (CNN) analysis based on real-time model confidence.

---

## 2. Dataset & Integrity Pipeline
To ensure scientific credibility and zero data leakage, PhishX uses a rigorous preprocessing pipeline:

- **Total Dataset Size:** ~632,508 unique URLs post-deduplication.
- **Deduplication:** Exact URL and near-duplicate removal (minor variations on the same path).
- **Domain-Level Split (Group-Split):** We split data such that **0% of domains** in the test set appear in the training set. This is a significantly harder and more realistic benchmark than random splitting.
- **Distribution:** ~50% Phishing / 50% Benign for balanced baseline training.
- **Test Size:** 129,872 samples (shuffled subset of 5,000–10,000 used for final scientific reporting).

---

## 3. Model Architecture (End-to-End)

### A. Semantic Engine (Transformer Backbone)
- **Architecture:** `distilbert-base-uncased` (Fine-tuned).
- **Function:** Captures linguistic relationships and semantic intent within the URL string.
- **Strength:** Excellent at catching brand-impersonation keywords (e.g., "bank-login-verify").

### B. Structural Engine (Character-Level CNN)
- **Architecture:** Dual-layer 1D-CNN with 32D embeddings.
- **Function:** Analyzes the raw character layout of the URL.
- **Strength:** Captures sub-word obfuscations, typosquatting, and structural anomalies (e.g., homoglyph attacks or IP-based URLs) that Transformers might miss.

### C. Adaptive Gating Network (The "Brain")
- **Mechanism:** A 3-layer MLP meta-classifier.
- **Inputs:** $[\mu_{transformer}, \sigma^2_{transformer}, \mu_{cnn}, \sigma^2_{cnn}, \text{Metadata Features}]$.
- **The Alpha ($\alpha$) Weight:** Represents the dynamic confidence the system has in the Transformer relative to the CNN.
- **Fusion Equation:** $P_{final} = \alpha \cdot P_{trans} + (1 - \alpha) \cdot P_{cnn}$.

### D. Bayesian Uncertainty (Monte Carlo Dropout)
- **Process:** During inference, the system performs **10 stochastic forward passes** with active dropout.
- **Benefit:** Instead of a single "guess," the model computes a **Mean (Risk)** and a **Variance (Uncertainty)**. High variance indicates an "Adversarial Trigger," signaling the system is unsure and requires manual review.

---

## 4. Training Strategy
We employ a **Multi-Stage Optimization** approach:

1.  **Stage 1 (Representation Learning):** Backbones (Transformer/CNN) are trained independently to become experts in their respective domains (Linguistic vs. Structural).
2.  **Stage 2 (Fusion Learning):** Backbones are **frozen**. The Gating Network is trained on a "Bayesian Feature Map" generated from the validation set. This prevents "catastrophic forgetting" and ensures the gating network focuses purely on reliability modeling.
3.  **Loss Function:** `BCEWithLogitsLoss` for numerical stability and calibration.

---

## 5. Evaluation & Validation
- **Scientific Benchmark:** Measured on 5,000 domains completely unseen during any training stage.
- **Primary Metrics:**
    - **AUPRC:** Superior for imbalanced scenarios.
    - **FPR (False Positive Rate):** Crucial for minimizing user friction.
    - **McNemar’s Test:** A statistical significance test performed to prove that our Adaptive Gating ($p < 0.05$) is mathematically better than static 50/50 or 70/30 weighting.

---

## 6. System Workflow (Inference Path)
1.  **QR Capture:** User scans QR via mobile/web interface.
2.  **Extraction:** QR decoder extracts the destination URL.
3.  **Feature Extraction:** Regex logic computes 8 metadata features (Length, Dots, TLD, etc.).
4.  **Bayesian Passes:** Transformer and CNN run 10 passes each with active dropout.
5.  **Gating Inference:** Gating network processes means/variances and decides the weight $\alpha$.
6.  **Fusion:** Weighted average yields the final **Risk Score**.
7.  **Output:** System returns **Prediction**, **Risk %**, and **Uncertainty Score**.

---

## 7. Results Interpretation
- **CNN Dominance:** We observed that the CNN often carries more weight ($\alpha < 0.5$) because character-level patterns are more stable indicators of quish-phishing than semantic keywords alone.
- **Calibration Result:** The uncertainty-aware model reduced Expected Calibration Error (ECE) significantly, meaning the predicted probability (e.g., 90%) actually aligns with the real-world hit rate.

---

## 8. Limitations & Future Work
- **Inference Overhead:** 10-pass MC Dropout increases latency (approx. 38ms vs. 8ms). Future versions may use **Deep Ensembles** or **Knowledge Distillation** to optimize.
- **Obfuscation Limits:** Highly polymorphic URLs or shortened redirects (bit.ly) require additional "crawler" modules to resolve the final destination before scanning.

---

## 9. Viva Defense / Interview Cheat Sheet
1.  **Why Multimodal?** To balance semantics (Transformer) with structure (CNN).
2.  **Why Bayesian?** To detect adversarial attacks where the model would otherwise be "confidently wrong."
3.  **What is Alpha?** A dynamic weight that adjusts the system's reliance in real-time.
4.  **Zero-Leakage?** No domain in the training set ever touches the test set.
5.  **Inference Stability?** Uses 10 passes to ensure the result isn't a "fluke" of the randomized dropout.
6.  **Primary Contribution?** The use of uncertainty-driven feedback to tune the gating network's reliance.
7.  **Detection Strength?** Statistically significant improvement over fixed ensembles ($p = 0.044$).
8.  **Real-world Readiness?** Low FPR (0.0015) in 1:100 imbalanced traffic tests.
