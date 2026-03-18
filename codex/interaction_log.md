# Interaction Log

Project: `/Users/benedicthalim/Desktop/UCL/Term 2/Predictive_analytics/group_project`

This file summarizes the user requests in this session and the corresponding actions taken.

## 1. Exploratory Data Analysis

### What you asked

You asked for a practical EDA for the predictive analytics workflow in Python using the provided dataset, including:
- an EDA plan,
- runnable Python code,
- data structure and type checks,
- missing values and summary statistics,
- target distribution and class imbalance,
- predictor-target relationships,
- appropriate plots,
- data quality issues and leakage risks,
- concise findings,
- and 3 preprocessing/model-selection implications.

### What I did

- Inspected the workspace and found the dataset `bank-additional-full.csv`.
- Inferred the modelling target as `y`.
- Wrote the EDA script `eda_bank_marketing.py`.
- Ran the script and generated outputs in `outputs/eda`.
- Reported evidence-based findings from the actual computed outputs.

### Files created

- `eda_bank_marketing.py`
- `outputs/eda/`

## 2. Baseline Model and Evaluation Harness

### What you asked

You asked for a sensible baseline model and evaluation harness in Python, including:
- a modelling plan,
- prediction type and metrics,
- reproducible split logic,
- a baseline model,
- preprocessing,
- reported evaluation results,
- and justification for why it is a baseline rather than an optimized model.

### What I did

- Assumed the same dataset and target as the EDA task.
- Built a baseline binary classification pipeline with:
  - stratified train/test split,
  - preprocessing for numeric and categorical variables,
  - `DummyClassifier` as a floor,
  - `LogisticRegression(class_weight="balanced")` as the baseline model.
- Dropped `duration` by default to reduce leakage risk.
- Wrote the script `baseline_bank_marketing.py`.
- Ran the script and reported the actual holdout metrics.
- Saved outputs under `outputs/baseline`.

### Files created

- `baseline_bank_marketing.py`
- `outputs/baseline/`

## 3. Improving the Logistic Regression Baseline

### What you asked

You asked for a valid, well-justified performance improvement over the existing logistic regression baseline, including:
- diagnosis of baseline weaknesses,
- up to 3 proposed improvements,
- a chosen approach,
- runnable Python code,
- a fair comparison against the original baseline,
- explanation of why it should help,
- and risk checks.

### What I did

- Reviewed the baseline harness.
- Benchmarked a small set of leakage-safe alternatives using the same split and metrics.
- Identified a tree-ensemble approach as the strongest bounded improvement.
- Implemented `improve_bank_marketing.py`.
- Compared:
  - `LogisticRegression(balanced)`
  - `RandomForest(balanced_subsample)`
- Kept the comparison fair by preserving:
  - the same dataset,
  - the same target,
  - the same `duration` exclusion,
  - the same split seed,
  - and the same evaluation metrics.
- Ran the script and saved results under `outputs/improved`.

### Files created

- `improve_bank_marketing.py`
- `outputs/improved/`

## 4. Leakage and Evaluation Review of `baseline_bank_marketing_v2.py`

### What you asked

You asked for a methodological review of `baseline_bank_marketing_v2.py`, specifically to:
- detect exact leakage or evaluation mistakes,
- explain exactly where they occur and why they are invalid,
- provide corrected runnable Python code without editing the file,
- show how corrected evaluation should be run,
- and explain how corrected results may differ.

### What I did

- Inspected `baseline_bank_marketing_v2.py`.
- Identified three concrete methodological issues:
  1. target leakage in `job_yes_rate`,
  2. preprocessing fitted before the train/test split,
  3. inclusion of `duration` by default.
- Ran a corrected evaluation separately without modifying the original file.
- Explained why the corrected results should be lower and more credible.

