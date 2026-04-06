# Individual Assignment 3

**DNSC 6330: Responsible Machine Learning**
George Washington University, School of Business
**Author:** Shanif Lakhani

---

## Overview

This assignment conducts a full disparate impact audit on the COMPAS recidivism risk scoring tool using the Broward County dataset. The analysis measures algorithmic bias across race and sex using industry-standard fairness metrics, replicates the solas-ai disparity library methodology, and produces a compliance-ready summary of findings.

---

## Dataset

**Source:** ProPublica COMPAS Analysis
**URL:** https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

**Cleaning pipeline (Lecture 01):**
- Removed records where `days_b_screening_arrest` outside [-30, 30]
- Removed records where `is_recid == -1`
- Removed records where `c_charge_degree == 'O'`
- Removed records where `score_text == 'N/A'`
- Final sample: **n = 6,172 defendants**

---

## Setup

```bash
pip install solas-ai
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import solas_disparity as sd
```

---

## Tasks

### Task 1 — AIR, ME, and SMD by Race and Sex

Computed Adverse Impact Ratio (AIR), Marginal Effect (ME), and Standardized Mean Difference (SMD) for race and sex separately. Results were cross-validated using the `solas_disparity` library to confirm identical outputs.

**Key results — Race (reference: Caucasian):**

| Group | Selection Rate | AIR | ME | Flag |
|---|---|---|---|---|
| Other | 0.064 | 0.605 | -0.042 | *** BELOW 0.80 |
| Hispanic | 0.092 | 0.871 | -0.014 | |
| Asian | 0.097 | 0.913 | -0.009 | |
| Caucasian | 0.106 | 1.000 | 0.000 | reference |
| African-American | 0.266 | 2.510 | +0.160 | |
| Native American | 0.364 | 3.429 | +0.258 | |

**Key results — Sex (reference: Male):**

| Group | Selection Rate | AIR | ME | Flag |
|---|---|---|---|---|
| Female | 0.129 | 0.647 | -0.070 | *** BELOW 0.80 |
| Male | 0.199 | 1.000 | 0.000 | reference |

The `solas_disparity` library confirmed the same affected groups: `Other` for race and `Female` for sex fell below the EEOC 0.80 threshold.

---

### Task 2 — Intersectional Analysis (Race × Sex)

Built intersectional subgroups combining race and sex. Only subgroups with n ≥ 30 were included.

**Results (reference: Caucasian / Male):**

| Subgroup | n | Selection Rate | AIR | Flag |
|---|---|---|---|---|
| Hispanic / Female | 82 | 0.012 | 0.114 | *** BELOW 0.80 |
| Other / Female | 58 | 0.017 | 0.162 | *** BELOW 0.80 |
| Other / Male | 285 | 0.074 | 0.690 | *** BELOW 0.80 |
| Caucasian / Female | 482 | 0.104 | 0.972 | |
| Caucasian / Male | 1621 | 0.107 | 1.000 | reference |
| Hispanic / Male | 427 | 0.108 | 1.009 | |
| African-American / Female | 549 | 0.179 | 1.673 | |
| African-American / Male | 2626 | 0.284 | 2.665 | |

**Worst subgroup:** Hispanic / Female — AIR = 0.114

Hispanic females are flagged as high risk at only 11.4% of the rate of Caucasian male defendants. This harm is completely invisible when examining race and sex independently, since Hispanic males have an AIR of 1.009 (essentially at parity). This directly illustrates the intersectionality principle from Lecture 03.

---

### Task 3 — FPR and FNR Disparity Analysis

Computed False Positive Rate (FPR) and False Negative Rate (FNR) by race, with statistical significance tested using two-proportion z-tests (Black vs. White).

**Error rates by race:**

| Race | n | FPR | FNR | Accuracy |
|---|---|---|---|---|
| African-American | 3175 | 0.139 | 0.618 | 0.610 |
| Hispanic | 509 | 0.063 | 0.857 | 0.642 |
| Caucasian | 2103 | 0.048 | 0.803 | 0.657 |
| Asian | 31 | 0.043 | 0.750 | 0.774 |
| Other | 343 | 0.014 | 0.847 | 0.685 |
| Native American | 11 | 0.167 | 0.400 | 0.727 |

**Statistical significance (Black vs. White):**

| Test | z-statistic | p-value | Result |
|---|---|---|---|
| FPR (Black vs White) | 8.154 | < 0.001 | Significant |
| FNR (Black vs White) | -9.276 | < 0.001 | Significant |

- **Delta FPR = +0.092** — Black defendants are falsely flagged at nearly 3× the rate of White defendants among those who would not have reoffended
- **Delta FNR = -0.185** — White defendants who would have reoffended are missed at a much higher rate than Black defendants

Both disparities are highly statistically significant and confirm the empirical pattern from Lecture 03: FPR_Black > FPR_White and FNR_White > FNR_Black.

---

### Task 4 — Publication-Quality Figure

Generated a grouped horizontal bar chart showing FPR and FNR by race with Caucasian as the reference baseline. African-American bars are highlighted in red to draw attention to the core disparity. Figure saved as `error_rate_disparity.png`.

---

### Task 5 — Compliance Memo

**To:** Office of the Regulator
**From:** Shanif Lakhani
**Date:** April 6, 2026

This memo summarises the findings of a bias audit conducted on the COMPAS recidivism risk scoring tool using the Broward County dataset (n = 6,172 defendants). The audit examined disparate impact across racial groups and sex using Adverse Impact Ratio (AIR), Marginal Effect (ME), Standardized Mean Difference (SMD), and error-rate disparity analysis.

The audit identified statistically and practically significant disparities across multiple metrics. On selection rate, African-American defendants were flagged as high risk at 2.51 times the rate of Caucasian defendants. Female defendants were selected at only 64.7% of the rate of male defendants (AIR = 0.647), falling below the EEOC 0.80 threshold.

Error-rate analysis revealed that Black defendants who did not reoffend were falsely labelled high risk at a rate of 13.9%, compared to 4.8% for White defendants (Delta FPR = +0.092, z = 8.154, p < 0.001). Conversely, White defendants who did reoffend were missed at a rate of 80.3% compared to 61.8% for Black defendants (Delta FNR = -0.185, z = -9.276, p < 0.001). Both disparities are highly statistically significant.

Intersectional analysis revealed that Hispanic female defendants were flagged at only 11.4% of the rate of Caucasian male defendants (AIR = 0.114), a harm entirely invisible when examining race and sex independently.

Base rate differences between groups are themselves products of historical policing disparities. The Impossibility Theorem confirms that simultaneous calibration and error-rate parity cannot be achieved under unequal base rates. We recommend per-group threshold adjustment, quarterly subgroup monitoring, and a proxy feature audit on variables such as prior arrests. Continued deployment without remediation creates material legal exposure under Title VII and ECOA.

---

## File Structure

```
├── Individual_Assignment_3.ipynb   # Main notebook
└── README.md                       # This file
```


