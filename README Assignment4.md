# Individual Assignment 4: From Accuracy to Accountability
### DNSC 6330 — Responsible Machine Learning

---

## Cumulative Audit Context

This assignment is the fourth step in a progressive audit:

| Assignment | Focus | Key Finding |
|---|---|---|
| **1** | EDA & Bias | African-American defendants are **1.45× more likely** to receive a high-risk COMPAS score than Caucasian defendants (controlling for charge, priors, recidivism). Black FPR = 0.367 vs. White FPR = 0.104 (Δ = +0.263). |
| **2** | Explainability (SHAP, LIME, DiCE) | `decile_score` and `priors_count` are the dominant global SHAP drivers. DiCE counterfactuals showed race/sex flips were not required to flip predictions but `priors_count` carries indirect racial bias risk as a proxy variable. |
| **3** | Fairness Metrics (AIR, ME, SMD) | African-American AIR = 2.51 (far over-selected). Female AIR = 0.647 (below 0.80 threshold). Intersectional worst-case: Hispanic female AIR = 0.114 vs. Caucasian male. FPR disparity confirmed statistically (z = 8.154, p < 0.001). |
| **4** | Robustness & Generalization | Do these disparities persist under drift, stress, and slicing? Does the model generalize or overfit to training period racial patterns? |

---

## Dataset

**COMPAS Recidivism Risk Scores** — ProPublica Broward County cohort  
Source: [github.com/propublica/compas-analysis](https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv)

- **Target:** `score_binary` — 1 = High Risk COMPAS score, 0 = Low Risk
- **Split:** 80% train / 20% test, stratified, `random_state=42`
- **n (after ProPublica filtering):** ~6,172 defendants

**Features:**

| Feature | Type | Notes |
|---------|------|-------|
| `priors_count` | Numeric | Prior convictions — dominant SHAP driver; racial proxy risk |
| `two_year_recid` | Numeric | Actual recidivism label |
| `age_factor` | Categorical | Age bracket (reference: 25–45) |
| `race_factor` | Categorical | Race (reference: Caucasian) |
| `gender_factor` | Categorical | Sex (reference: Male) |
| `crime_factor` | Categorical | Charge degree - Felony or Misdemeanor |

---

## Models

| Model | Role | Key Properties |
|-------|------|----------------|
| **Logistic Regression** | Interpretable baseline | Lower variance, smaller generalization gap, auditable coefficients, preferred for regulated deployment |
| **Gradient-Boosted Tree** | Higher-capacity model | 200 estimators, max depth 4:- higher peak AUC but greater overfitting risk and potential to encode training period racial correlations |

Both use `sklearn` pipelines with `StandardScaler` (numeric) and `OneHotEncoder` (categorical).

---

### Lecture Foundation 

Carried forward from Prof. Akinwumi's live-coding notebooks across all four lectures:

**Lectures 01–03:**
- COMPAS data loading, ProPublica filtering, EDA
- Logistic regression for racial bias; confusion matrices by race; FPR/FNR disparity
- Train/test split; LR and GBT sklearn pipelines
- LIME explanations, SHAP beeswarm and waterfall plots, DiCE counterfactuals
- AIR, Marginal Effect, SMD by race and sex; intersectional subgroup analysis (race × gender)

**Lecture 04 Live Coding:**
- Helper functions: `psi_numeric`, `mmd_rbf`, `evaluate_classifier`, `pairwise_swap_shift`, `slice_metrics`, `stress_test_priors`, `plot_ice_numeric`, `global_sensitivity_index`
- Distribution drift: PSI, KS, MMD²
- Generalization gaps: AUC, accuracy, Brier, log loss
- Counterfactual swap sensitivity
- Slice-based evaluation by race, gender, age, crime type
- Stress testing and ICE curves for `priors_count`
- Compact live-coding summary table

### Homework Extension 

| Cell Group | Content |
|---|---|
| Header | Assignment context; cumulative findings table from Assignments 1–3 |
| **Part A** | PSI bar chart, KS chart, score distribution plots + interpretation connecting drift risk to racial proxy features |
| **Part B** | Generalization gap bar chart (AUC, Accuracy, Brier, Log Loss) + interpretation connecting GBT overfitting to Assignment 2 SHAP instability |
| **Part C** | Counterfactual swap bar chart with 0.05 concern threshold + interpretation tying shifts to Assignment 3 AIR = 2.51 finding |
| **Part D** | Stress test line chart, ICE curves, sensitivity index chart + interpretation explicitly naming `priors_count` proxy-discrimination risk |
| **Part E** | Grouped AUC/FPR/FNR charts by race + **FPR comparison vs. Assignment 1 baseline** + interpretation invoking Impossibility Theorem |
| **Memo** | Full governance memo connecting all four assignments into a single audit narrative with 5 concrete recommendations |

---

## Key Findings

| Part | Finding | Connection to Prior Work |
|---|---|---|
| **A — Drift** | All PSI < 0.10; MMD² ≈ 0; score distributions aligned | `priors_count` and `age` are racial proxies (A2),  demographic shifts in production would manifest as PSI alerts on these features |
| **B — Generalization** | GBT has larger train-test AUC gap than LR | Consistent with GBT's less stable SHAP attributions in A2; may encode training-period racial correlations |
| **C — Spurious Correlations** | Race and gender swaps produce non-zero prediction shifts | Consistent with A3 AIR = 2.51 for African-American defendants; model has encoded racial sensitivity via `priors_count` |
| **D — Robustness** | High $V_j$ for `priors_count`; heterogeneous ICE effects | `priors_count` is the top SHAP feature (A2) and correlates with race (A1/A3),  creates legal exposure under ECOA |
| **E — Slice Evaluation** | FPR disparities by race persist; Black FPR > White FPR in both models | Directly replicates A1 finding (Δ FPR = +0.263) and A3 AIR finding; comparison chart shows whether current models improve or worsen the baseline |

---

## Governance Recommendations (from Memo)

1. **Adopt Logistic Regression over GBT** for regulated deployment — smaller generalization gap, more stable SHAP attributions, lower risk of encoding training-period racial correlations
2. **Conduct a Less Discriminatory Alternative (LDA) analysis** — train a model excluding `priors_count` and test whether racial AIR improves while AUC remains acceptable
3. **Implement per-group threshold recalibration** to reduce FPR disparities, with explicit documentation of the FNR tradeoff per the Impossibility Theorem
4. **Deploy monthly monitoring** tracking PSI for all features, AUC by race slice, and FPR/FNR disparity with automated alerts
5. **Require MRM documentation for `priors_count`** acknowledging its proxy-discrimination risk, the A1–A3 evidence of racial correlation, and the governance rationale for retaining it

---

## How to Run

1. Open `Individual_Assignment_4.ipynb` in **Google Colab**
2. Run all cells top to bottom (`Runtime → Run all`)
3. Install required packages at the top of the notebook:
   ```
   !pip install lime shap dice-ml solas-ai
   ```
4. No local data files required — the COMPAS dataset loads directly from the ProPublica GitHub URL

---

## File Structure

```
├── Individual_Assignment_4.ipynb    ← Main submission notebook
└── README_Assignment4.md            ← This file
```
