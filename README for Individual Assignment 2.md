# DNSC 6330: Responsible Machine Learning
## Individual Assignment 2

**Author:** Shanif Lakhani  
**Course:** DNSC 6330 — Responsible Machine Learning  
**Instructor:** Michael Akinwumi

---

## Purpose of the Analysis

This project builds on the COMPAS recidivism dataset used in Lectures 01 and 02. The COMPAS tool is a risk assessment algorithm used in the US criminal justice system to predict whether a defendant will reoffend within two years.

The goal of this analysis is to go beyond just measuring model performance, we want to understand **why** the model makes the predictions it does. We do this using three explainability methods taught in Lecture 02:

1. **SHAP** — shows which features push each prediction up or down. 
2. **LIME** — fits a simple local model to explain one prediction at a time. 
3. **DiCE** — generates counterfactuals showing the smallest changes needed to flip a prediction. 

We also check whether the model's explanations differ across racial groups, and whether any counterfactuals require changes to immutable features like race or sex — which would raise serious fairness concerns.

---

## Python Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, preprocessing, evaluation |
| `shap` | SHAP values — beeswarm and waterfall plots |
| `lime` | LIME local explanations |
| `dice-ml` | DiCE counterfactual explanations |
| `matplotlib` | Plotting |

Install all required libraries by running:

```bash
pip install shap lime dice-ml
```

---

## Repository Structure

```
├── Individual_Assignment_2.ipynb   # Notebook
└── README.md                       # This file
```

---

## Instructions for Reproducing the Results

### Step 1 — Open the notebook in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `Individual_Assignment_2.ipynb`

### Step 2 — Install required libraries

Run the following cell at the top of the notebook:

```python
!pip install shap lime dice-ml --quiet
```

### Step 3 — Run all cells in order

Go to **Runtime → Run all** to execute every cell from top to bottom.

The dataset loads automatically from ProPublica's public GitHub — no manual download is needed:

```python
URL = (
    'https://raw.githubusercontent.com/propublica/compas-analysis/'
    'master/compas-scores-two-years.csv'
)
```

### Step 4 — Expected outputs

Running all cells will produce the following outputs:

| Step | Output |
|---|---|
| Data loading | Raw shape: (7214, 53), Cleaned shape: (6172, 53) |
| Train/test split | Train: (4937, 9), Test: (1235, 9) |
| Model fitting | Logistic Regression + Gradient-Boosted Tree fitted |
| Q1 — SHAP | Beeswarm summary plot + waterfall plots for 4 individuals |
| Q2 — LIME | Feature attributions for Black and White defendants + comparison with SHAP |
| Q3 — DiCE | Counterfactuals for 4 individuals + immutable feature audit |
| Q4 — Memo | 300-word governance memo in markdown |

---

## Key Findings

- `decile_score` and `priors_count` are the strongest global predictors of recidivism risk. 
- All four DiCE counterfactuals passed the immutable feature audit. Race and sex did **not** need to change to flip any prediction. 
- The features that needed to change were `age`, `priors_count`, and `c_charge_degree`. 
- LIME and SHAP agreed on the most important features but diverged on exact rankings, suggesting explanation instability that should be considered in any legal or governance context. 

---

## Notes

- Random seed is set to `42` throughout for reproducibility. 
- The GBT model uses `n_estimators=200`, `max_depth=4` as specified in Lecture 02, Slide 48.
- SHAP uses `check_additivity=False` to handle minor numerical precision differences in the GBT model. 
