# File Index

Complete file-by-file index of all agent outputs in the repository.

## Codex

| Path | Description | Task |
|------|-------------|------|
| `codex/bank-additional-full.csv` | UCI Bank Marketing dataset (41,188 rows, 21 columns) | Support |
| `codex/eda_bank_marketing.py` | Exploratory data analysis script with Mann-Whitney tests, lift analysis, and leakage screening | EDA |
| `codex/baseline_bank_marketing.py` | Initial LR baseline (includes `duration`, AUROC=0.9437) | Baseline |
| `codex/baseline_bank_marketing_v2.py` | Corrected LR baseline without `duration` (AUROC=0.8006) | Baseline |
| `codex/improve_bank_marketing.py` | Random Forest improvement script (AUROC=0.8063, PR-AUC=0.4905) | Improved |
| `codex/leakage_analysis_baseline_codex.ipynb` | Jupyter notebook auditing baseline for data leakage issues | Leakage |
| `codex/interaction_log.md` | Session interaction log documenting prompts and agent responses | Support |
| `codex/output_eda/screening_summary_table.csv` | Feature screening summary with statistical tests and leakage flags | EDA |
| `codex/output_eda/numeric_summary.csv` | Descriptive statistics for numeric features | EDA |
| `codex/output_eda/structure_table.csv` | Dataset structure overview (dtypes, unique counts) | EDA |
| `codex/output_eda/target_distribution.csv` | Target variable class distribution | EDA |
| `codex/output_eda/missing_and_unknown_summary.png` | Bar chart of missing and unknown values per feature | EDA |
| `codex/output_eda/target_distribution.png` | Target variable distribution plot | EDA |
| `codex/output_eda/categorical_target_relationships.csv` | Subscription rates by categorical feature levels | EDA |
| `codex/output_eda/categorical_target_rates.png` | Subscription rate bar charts for categorical features | EDA |
| `codex/output_eda/numeric_target_relationships.csv` | Numeric feature distributions by target class | EDA |
| `codex/output_eda/numeric_vs_target.png` | Boxplots of numeric features split by target | EDA |
| `codex/output_eda/numeric_correlation_heatmap.png` | Correlation heatmap for numeric features | EDA |
| `codex/output_eda/data_quality_issues.csv` | Summary of data quality issues found | EDA |
| `codex/output_eda/potential_leakage_risks.csv` | Features flagged as potential leakage risks | EDA |
| `codex/output_eda/unknown_token_table.csv` | Counts and proportions of "unknown" tokens per feature | EDA |
| `codex/output_baseline/baseline_results.csv` | Baseline model performance metrics | Baseline |
| `codex/output_baseline/logistic_classification_report.csv` | Per-class precision, recall, F1 for baseline LR | Baseline |
| `codex/output_baseline/logistic_confusion_matrix.png` | Confusion matrix heatmap for baseline LR | Baseline |
| `codex/output_improved/baseline_vs_improved_results.csv` | Side-by-side comparison of baseline vs improved metrics | Improved |
| `codex/output_improved/classification_reports.csv` | Per-class metrics for improved RF model | Improved |
| `codex/output_improved/metric_comparison.png` | Bar chart comparing baseline and improved model metrics | Improved |
| `codex/output_improved/temp` | Temporary files from model training | Support |

## Claude

| Path | Description | Task |
|------|-------------|------|
| `Claude/bank-additional-full.csv` | UCI Bank Marketing dataset (41,188 rows, 21 columns) | Support |
| `Claude/eda.py` | Exploratory data analysis script with correlation analysis, skewness, and IQR outlier detection | EDA |
| `Claude/baseline_model.py` | Logistic Regression baseline (AUROC=0.8009, PR-AUC=0.4603) | Baseline |
| `Claude/baseline_model_corrected.py` | Corrected baseline after leakage audit | Baseline |
| `Claude/improved_model.py` | HistGradientBoosting improvement with CV-tuned threshold (AUROC=0.8163) | Improved |
| `Claude/leakage_analysis.ipynb` | Jupyter notebook auditing baseline for data leakage issues | Leakage |
| `Claude/session_log.txt` | Session interaction log documenting prompts and agent responses | Support |
| `Claude/output_eda/eda_summary_table.csv` | Feature summary statistics table | EDA |
| `Claude/output_eda/fig_target_distribution.png` | Target variable distribution plot | EDA |
| `Claude/output_eda/fig_numeric_distributions.png` | Histograms of numeric feature distributions | EDA |
| `Claude/output_eda/fig_categorical_distributions.png` | Bar charts of categorical feature distributions | EDA |
| `Claude/output_eda/fig_sub_rate_by_categorical.png` | Subscription rates by categorical feature levels | EDA |
| `Claude/output_eda/fig_boxplots_by_target.png` | Boxplots of numeric features split by target | EDA |
| `Claude/output_eda/fig_correlation_heatmap.png` | Correlation heatmap for numeric features | EDA |
| `Claude/output_eda/fig_feature_correlation_with_target.png` | Bar chart of feature correlations with target variable | EDA |
| `Claude/output_baseline/baseline_results_summary.csv` | Baseline model performance metrics | Baseline |
| `Claude/output_baseline/baseline_corrected_results.csv` | Corrected baseline metrics after leakage fix | Baseline |
| `Claude/output_baseline/fig_baseline_roc_pr.png` | ROC and PR curves for initial baseline | Baseline |
| `Claude/output_baseline/fig_baseline_lr_coefficients.png` | Logistic Regression coefficient magnitudes | Baseline |
| `Claude/output_baseline/fig_baseline_corrected_roc_pr.png` | ROC and PR curves for corrected baseline | Baseline |
| `Claude/output_improved/improved_results_summary.csv` | Improved model performance metrics | Improved |
| `Claude/output_improved/fig_improved_roc_pr.png` | ROC and PR curves for improved model | Improved |
| `Claude/output_improved/fig_improved_feature_importance.png` | Feature importance bar chart for HGBC model | Improved |
| `Claude/output_improved/fig_baseline_vs_improved.png` | Side-by-side metric comparison of baseline vs improved | Improved |
| `Claude/output_leakage/fig_leakage_comparison.png` | Metric comparison between leaked and corrected models | Leakage |
| `Claude/output_leakage/fig_leakage_roc_pr_comparison.png` | ROC and PR curves comparing leaked vs corrected | Leakage |
| `Claude/output_leakage/fig_leakage_coefficients.png` | Coefficient comparison between leaked and corrected LR | Leakage |

## Antigravity

| Path | Description | Task |
|------|-------------|------|
| `Antigravity/Antigravity_EDA.ipynb` | Exploratory data analysis with boxplots, countplots, distribution analysis, and leakage risk identification | EDA |
| `Antigravity/Antigravity_baseline_model.ipynb` | Logistic Regression baseline with Pipeline, AUROC=0.8010, PR-AUC=0.4601 | Baseline |
| `Antigravity/Antigravity_performance improvement.ipynb` | Random Forest improvement, AUROC=0.8128, PR-AUC=0.4912, F1=0.50 | Improved |
| `Antigravity/Antigravity_data_leakage.ipynb` | Leakage audit finding 3 issues, leaked AUROC=0.9438 vs corrected | Leakage |
| `Antigravity/interactions_log_Antigravity.md` | Session interaction log | Support |
| `Antigravity/temp` | Temporary files | Support |
