# Comparison Summary

## 1. Task-by-Task Comparison

| Task | Codex | Claude | Antigravity |
|------|-------|--------|-------------|
| **EDA** | Comprehensive screening of all 21 features with Mann-Whitney U tests, lift analysis, and leakage risk flagging. 5 PNGs + 9 CSVs output. | Analysed 10 numeric + 10 categorical features with correlation analysis, skewness, and IQR outlier detection. 7 PNGs + 1 CSV output. | Boxplots for numeric-target relationships, countplots for categoricals, flagged `duration` leakage risk, `pdays=999` sentinel, unknown values, and class imbalance. All inline in a single Jupyter notebook. |
| **Baseline** | LR (balanced) AUROC=0.9437 (WITH `duration`), Dummy floor=0.50. Note: `duration` was included in the initial version. | LR (balanced) AUROC=0.8009, PR-AUC=0.4603, F1=0.468. `duration` excluded from the start. | LR (balanced) AUROC=0.8010, PR-AUC=0.4601, F1=0.47. `duration` excluded. CV ROC-AUC=0.7896, CV PR-AUC=0.4438. Created binary `contacted_previously` from pdays. Used Pipeline to prevent preprocessing leakage. |
| **Improvement** | RF (balanced_subsample) AUROC=0.8063, PR-AUC=0.4905, F1=0.521 (without `duration`). | HistGradientBoosting AUROC=0.8163, PR-AUC=0.4868, F1=0.529 with CV-tuned threshold (0.634). | RF (n_estimators=200, max_depth=10, min_samples_leaf=4, balanced) AUROC=0.8128, PR-AUC=0.4912, F1=0.50. No threshold tuning. |
| **Leakage Audit** | Found 3 issues: `job_yes_rate` target encoding, `fit_transform` before split, `duration` included. | Found 3 issues: `job_yes_rate` target encoding, `fit_transform` before split, `duration` not dropped. | Found 3 issues: `job_yes_rate` target encoding on full data, preprocessing (StandardScaler) fit before split, CV contamination from global preprocessing. Leaked metrics: Test ROC-AUC=0.9438, Test PR-AUC=0.6222. Corrected via proper Pipeline implementation. |

## 2. Baseline Convergence

All three agents independently produced nearly identical Logistic Regression baselines when `duration` is excluded:

| Agent | AUROC | PR-AUC | F1 (yes) | CV AUROC |
|-------|-------|--------|----------|----------|
| Codex (v2, no duration) | 0.8006 | 0.4597 | 0.466 | ~0.79 |
| Claude | 0.8009 | 0.4603 | 0.468 | 0.7896 |
| Antigravity | 0.8010 | 0.4601 | 0.47 | 0.7896 |

This convergence (AUROC within 0.0004) validates that all three agents correctly implemented the same evaluation protocol with the same random seed and stratified split.

## 3. Improvement Strategies Compared

| Aspect | Codex | Claude | Antigravity |
|--------|-------|--------|-------------|
| Improved model | Random Forest (balanced_subsample) | HistGradientBoosting (balanced) | Random Forest (balanced, max_depth=10) |
| AUROC gain | +0.0057 | +0.0154 | +0.0118 |
| PR-AUC gain | +0.0308 | +0.0265 | +0.0311 |
| F1 gain | +0.055 | +0.061 | +0.03 |
| Threshold tuning | No | Yes (CV-based, 0.634) | No |
| Feature engineering | duration removed | pdays_clean, education_ordinal, high_campaign | contacted_previously from pdays |

## 4. Leakage Audit Convergence

All three agents independently identified the same core leakage issues when auditing a flawed baseline:

| Issue | Codex | Claude | Antigravity |
|-------|-------|--------|-------------|
| `job_yes_rate` target encoding on full data | Found | Found | Found |
| `fit_transform` / StandardScaler before split | Found | Found | Found |
| `duration` included as feature | Found | Found | Found (in EDA, excluded from start) |

This independent convergence on the same 3 issues demonstrates that all three agents can reliably detect common ML pitfalls.

## 5. Evaluation Criteria Matrix

| Criterion | Codex | Claude | Antigravity |
|-----------|-------|--------|-------------|
| Correctness | Run `eda_bank_marketing.py`, `baseline_bank_marketing.py`, `improve_bank_marketing.py` | Run `eda.py`, `baseline_model.py`, `improved_model.py` | Open all 4 `.ipynb` notebooks |
| Statistical validity | `leakage_analysis_baseline_codex.ipynb` | `leakage_analysis.ipynb`, `baseline_model_corrected.py` | `Antigravity_data_leakage.ipynb` |
| Reproducibility | `interaction_log.md` | `session_log.txt` | `interactions_log_Antigravity.md` |
| Code quality | All `.py` files | All `.py` files | All `.ipynb` notebooks |
| Efficiency | `interaction_log.md` (iteration count) | `session_log.txt` (iteration count) | `interactions_log_Antigravity.md` |

## 6. Key Differences Between Agents

- **Duration handling**: Codex initially included `duration` in its baseline (AUROC=0.94), then corrected it. Claude and Antigravity both excluded `duration` from the very start.
- **Baseline convergence**: All three agents used Logistic Regression (balanced) as the baseline and converged to AUROC ~0.80.
- **Improvement approach**: Codex and Antigravity both chose Random Forest for improvement. Claude chose HistGradientBoosting with CV-based threshold tuning, achieving the highest F1 (0.529).
- **Hyperparameter strategy**: Antigravity's RF used explicit hyperparameters (max_depth=10, min_samples_leaf=4) while Codex used balanced_subsample with defaults.
- **Leakage detection**: All three agents' leakage audits identified the same 3 core issues independently, showing reliable convergence.
- **Output format**: Codex used `.py` scripts with CSV/PNG output folders. Claude used `.py` scripts + one `.ipynb` for leakage with PNG output folders. Antigravity used Jupyter notebooks throughout with no separate output folders.
- **Output volume**: Codex produced the most tabular outputs (9 CSVs for EDA alone). Claude produced the most visualisations (7 PNGs for EDA). Antigravity kept outputs inline in notebooks.
