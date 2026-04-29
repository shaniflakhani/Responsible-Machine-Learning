# Individual Homework 05 — ML Security & Abuse Pathways
**DNSC 6330 · George Washington University School of Business**  

---

## Overview

This notebook extends `Individual Assignment 5.ipynb` to conduct a full adversarial security audit of the **ProPublica COMPAS recidivism dataset**. All three attack classes from the lecture. Evasion, poisoning, and membership inference are extended and deepened across four homework parts. All attack pipelines are implemented from scratch using `scikit-learn`, `numpy`, `pandas`, and `matplotlib`. No external adversarial ML library is used.

**Dataset:** [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv)  
6,172 records after filtering · 7 features · 45.5% recidivism rate

---

## Notebook Structure

|Section | Description |
|---------|---------|-------------|
|Setup | Data loading, preprocessing, LR + GBT training, clean-model fairness baseline |
|Part A (Lecture) | PGD evasion attack on LR — FPR by race, AIR sweep |
|Part B (Lecture) | Label-flip poisoning loop on LR — AUC and AIR degradation |
|Part C (Lecture) | Shadow-model membership inference on GBT — ROC curve |
|Part 1 | `pgd_gbt()` — numerical PGD for GBT via finite differences |
|Part 1 | PGD sweep across ε ∈ {0.25, 0.5, 1.0, 2.0} on both LR and GBT |
|Part 1 | 2×3 visualization — FPR by race, AIR, ΔFPR for LR vs. GBT |
|Part 2 | Poisoning sweep with `target_race='Caucasian'` |
|Part 2 | Degradation curves for both target-race variants + stealth zone |
|Part 2 | PSI drift monitor — feature-level PSI for both attack variants |
|Part 3 | Shadow-model MI attack on LR + confidence-gap histograms |
|Part 3 | Generalization gap vs. MI AUC — two-model comparison |
|Part 3 | L₂ regularization sweep (C ∈ {0.01, 0.1, 1.0, 10.0}) — MI AUC vs. C |
|Part 4 | Cross-part evidence summary for reflection |

---

## Requirements

```
scikit-learn
numpy
pandas
matplotlib
```

Run in **Google Colab** or any Python 3.8+ environment. No GPU required.  
The dataset is loaded automatically from the ProPublica GitHub URL, no local file needed.

---

## Key Results

### Part 1 — PGD Evasion Audit

| ε | LR FPR AA | LR FPR CA | LR AIR | GBT FPR AA | GBT FPR CA | GBT AIR |
|---|-----------|-----------|--------|------------|------------|---------|
| 0.00 | 0.281 | 0.143 | 1.961 | 0.317 | 0.178 | 1.782 |
| 0.25 | 0.569 | 0.370 | 1.535 | 0.317 | 0.178 | 1.782 |
| 0.50 | 0.791 | 0.560 | 1.411 | 0.317 | 0.178 | 1.782 |
| 1.00 | 0.978 | 0.884 | 1.106 | 0.317 | 0.178 | 1.782 |
| 2.00 | 1.000 | 1.000 | 1.000 | 0.317 | 0.178 | 1.782 |

- **LR AIR crosses 0.80:** Does not fall below 0.80 within tested range
- **GBT AIR crosses 0.80:** Does not fall below 0.80 within tested range
- **Finding:** LR is highly vulnerable. FPR_AA saturates to 1.000 at ε = 2.0. GBT is completely resistant at all ε values due to its piecewise constant decision boundary.

### Part 2 — Poisoning Loop with Fairness Monitoring

| Poison Rate | AA Target AUC | AA Target AIR | CA Target AUC | CA Target AIR |
|-------------|--------------|--------------|--------------|--------------|
| 0% | 0.735 | 1.961 | 0.735 | 1.961 |
| 15% | 0.732 | 2.134 | 0.733 | 1.913 |
| 30% | 0.731 | 3.010 | 0.732 | 1.940 |

- **Stealth zone (AA target):** [0%, 30%] AUC drop never exceeded −0.004 pp while AIR rose to 3.010
- **Stealth zone (CA target):** [0%, 30%] AUC drop never exceeded −0.003 pp while AIR declined to 1.842
- **PSI monitor:** PSI = 0.00000 on every feature at all poison rates, label flip poisoning is **invisible** to feature level drift monitoring

### Part 3 — Membership Inference Depth

| Model | Train AUC | Test AUC | Gen Gap | MI AUC |
|-------|-----------|----------|---------|--------|
| Logistic Regression | 0.727 | 0.735 | −0.008 | 0.497 |
| Gradient Boosted Tree | 0.798 | 0.718 | +0.080 | 0.500 |

- Both models produce MI AUC ≈ random (0.50) no exploitable membership leakage
- Gen gap hypothesis: directionally confirmed (GBT has larger gap and higher MI AUC) but effect is negligible on this tabular dataset

**L₂ Regularization Sweep:**

| C | MI AUC | Test AUC | AIR |
|---|--------|----------|-----|
| 0.01 | 0.499 | 0.732 | 1.966 |
| 0.10 | 0.505 | 0.734 | 1.946 |
| 1.00 | 0.497 | 0.735 | 1.961 |
| 10.00 | 0.499 | 0.735 | 1.961 |

- **Optimal:** C = 1.0 — highest test AUC (0.735), stable AIR (1.961), MI AUC at random. No privacy utility fairness argument supports deviating from the default.

---

## Part 4 — Reflection Summary

**Highest risk finding:** Label flip poisoning targeting African-American defendants. At 15% poison rate (172 flipped labels), AIR rose from 1.961 to 2.134 with only −0.003 pp AUC loss completely invisible to standard monitoring.

**Proactive mitigation:** Cryptographic label provenance tracking + per-demographic positive label rate monitoring on every training batch. A monitor calibrated to flag deviations > ±1 pp would detect the attack at 2% poison rate (23 flips).

**Reactive mitigation:** Production AIR alert dashboard firing when AIR exits [0.80, 1.25]. Would have flagged the AA-target attack at 15% poison rate before AIR reached 3.010.

**Disparate impact of mitigations:** Both controls are symmetric across groups. Neither addresses the pre-existing baseline disparity (AIR = 1.961); a threshold calibration step should precede deployment.

---
