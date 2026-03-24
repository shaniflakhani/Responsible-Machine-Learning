# COMPAS Analysis - Python

This is a Python version of the COMPAS recidivism analysis originally written in R. The goal is to reproduce the same results using Python.

## What this project does

The analysis looks at the COMPAS risk scoring system used in the US criminal justice system. It checks whether the scores are biased against certain racial groups. The dataset comes from ProPublica's investigation.

The script does the following:
- Loads the COMPAS dataset from GitHub
- Cleans and filters the data
- Looks at demographic breakdowns (race, age, sex)
- Builds a logistic regression model to predict high/low risk scores
- Checks how accurate the model is for different racial groups

## Libraries used

- pandas
- numpy
- matplotlib
- statsmodels

You can install them by running:

```
pip install pandas numpy matplotlib statsmodels
```

## How to run

1. Download the file `Individual Assignment 1.py`
2. Open your terminal or command prompt
3. Run this command:

```
python Individual Assignment 1.py
```

The script will automatically download the dataset from the internet so you do not need to download any data files separately.

## Expected output

- Printed tables showing demographic breakdowns
- A bar chart comparing decile scores for Black and White defendants
- Logistic regression summary table
- Confusion matrix and accuracy metrics overall and by race
