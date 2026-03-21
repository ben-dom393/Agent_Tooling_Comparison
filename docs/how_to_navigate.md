# How to Navigate This Repository

A quick-reference guide for assessing the project.



One key file per task, per agent:

- **EDA**: [`codex/output_eda/screening_summary_table.csv`](../codex/output_eda/screening_summary_table.csv) vs [`Claude/output_eda/eda_summary_table.csv`](../Claude/output_eda/eda_summary_table.csv) vs [`Antigravity/Antigravity_EDA.ipynb`](../Antigravity/Antigravity_EDA.ipynb) (all outputs inline)
- **Baseline**: [`codex/output_baseline/baseline_results.csv`](../codex/output_baseline/baseline_results.csv) vs [`Claude/output_baseline/baseline_corrected_results.csv`](../Claude/output_baseline/baseline_corrected_results.csv) vs [`Antigravity/Antigravity_baseline_model.ipynb`](../Antigravity/Antigravity_baseline_model.ipynb) (scroll to classification report and metrics cells)
- **Improvement**: [`codex/output_improved/baseline_vs_improved_results.csv`](../codex/output_improved/baseline_vs_improved_results.csv) vs [`Claude/output_improved/improved_results_summary.csv`](../Claude/output_improved/improved_results_summary.csv) vs [`Antigravity/Antigravity_performance improvement.ipynb`](../Antigravity/Antigravity_performance%20improvement.ipynb) (scroll to comparison table)
- **Leakage Audit**: [`codex/leakage_analysis_baseline_codex.ipynb`](../codex/leakage_analysis_baseline_codex.ipynb) vs [`Claude/leakage_analysis.ipynb`](../Claude/leakage_analysis.ipynb) vs [`Antigravity/Antigravity_data_leakage.ipynb`](../Antigravity/Antigravity_data_leakage.ipynb)
- **Process Log**: [`codex/interaction_log.md`](../codex/interaction_log.md) vs [`Claude/session_log.txt`](../Claude/session_log.txt) vs [`Antigravity/interactions_log_Antigravity.md`](../Antigravity/interactions_log_Antigravity.md)
- **Cross-agent convergence**: See [`docs/comparison_summary.md`](comparison_summary.md) Section 2 (Baseline Convergence) — all three agents hit AUROC=0.80 independently

## Deep Dive by Task

### EDA

| Agent | Script / Notebook | Outputs |
|-------|-------------------|---------|
| Codex | [`codex/eda_bank_marketing.py`](../codex/eda_bank_marketing.py) | [`codex/output_eda/`](../codex/output_eda/) — 5 PNGs + 9 CSVs |
| Claude | [`Claude/eda.py`](../Claude/eda.py) | [`Claude/output_eda/`](../Claude/output_eda/) — 7 PNGs + 1 CSV |
| Antigravity | [`Antigravity/Antigravity_EDA.ipynb`](../Antigravity/Antigravity_EDA.ipynb) | Inline in notebook |

### Baseline Model

| Agent | Script / Notebook | Outputs |
|-------|-------------------|---------|
| Codex | [`codex/baseline_bank_marketing.py`](../codex/baseline_bank_marketing.py), [`codex/baseline_bank_marketing_v2.py`](../codex/baseline_bank_marketing_v2.py) | [`codex/output_baseline/`](../codex/output_baseline/) — 2 CSVs + 1 PNG |
| Claude | [`Claude/baseline_model.py`](../Claude/baseline_model.py), [`Claude/baseline_model_corrected.py`](../Claude/baseline_model_corrected.py) | [`Claude/output_baseline/`](../Claude/output_baseline/) — 2 CSVs + 3 PNGs |
| Antigravity | [`Antigravity/Antigravity_baseline_model.ipynb`](../Antigravity/Antigravity_baseline_model.ipynb) | Inline in notebook |

### Improved Model

| Agent | Script / Notebook | Outputs |
|-------|-------------------|---------|
| Codex | [`codex/improve_bank_marketing.py`](../codex/improve_bank_marketing.py) | [`codex/output_improved/`](../codex/output_improved/) — 2 CSVs + 1 PNG |
| Claude | [`Claude/improved_model.py`](../Claude/improved_model.py) | [`Claude/output_improved/`](../Claude/output_improved/) — 1 CSV + 3 PNGs |
| Antigravity | [`Antigravity/Antigravity_performance improvement.ipynb`](../Antigravity/Antigravity_performance%20improvement.ipynb) | Inline in notebook |

### Leakage Audit

| Agent | Notebook | Key Finding |
|-------|----------|-------------|
| Codex | [`codex/leakage_analysis_baseline_codex.ipynb`](../codex/leakage_analysis_baseline_codex.ipynb) | 3 leakage issues: target encoding, fit before split, duration |
| Claude | [`Claude/leakage_analysis.ipynb`](../Claude/leakage_analysis.ipynb) | 3 leakage issues: target encoding, fit before split, duration |
| Antigravity | [`Antigravity/Antigravity_data_leakage.ipynb`](../Antigravity/Antigravity_data_leakage.ipynb) | 3 leakage issues: target encoding, global preprocessing, CV contamination |
