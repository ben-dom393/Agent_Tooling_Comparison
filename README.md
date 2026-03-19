# Agent Tooling Comparison — Predictive Analytics Benchmark

Three AI coding agents — **OpenAI Codex**, **Anthropic Claude**, and **Antigravity** — independently tackle the same end-to-end machine-learning pipeline on the UCI Bank Marketing dataset. Each agent performs exploratory data analysis, builds a logistic-regression baseline, improves performance with a tree-based model, and audits a flawed script for data leakage. We then compare the agents across correctness, statistical validity, code quality, reproducibility, and efficiency.

## Repository Structure

| Folder | Contents |
|--------|----------|
| `codex/` | OpenAI Codex agent outputs — `.py` scripts, Jupyter leakage notebook, CSV/PNG output folders, interaction log |
| `Claude/` | Anthropic Claude agent outputs — `.py` scripts, Jupyter leakage notebook, CSV/PNG output folders, session log |
| `Antigravity/` | Antigravity agent outputs — Jupyter notebooks for all four tasks, interaction log |
| `docs/` | Supporting documentation — comparison summary, file index, navigation guide |

## Quick Navigation

| Task | Codex | Claude | Antigravity |
|------|-------|--------|-------------|
| EDA | [eda_bank_marketing.py](codex/eda_bank_marketing.py) | [eda.py](Claude/eda.py) | [Antigravity_EDA.ipynb](Antigravity/Antigravity_EDA.ipynb) |
| Baseline Model | [baseline_bank_marketing.py](codex/baseline_bank_marketing.py) | [baseline_model.py](Claude/baseline_model.py) | [Antigravity_baseline_model.ipynb](Antigravity/Antigravity_baseline_model.ipynb) |
| Improved Model | [improve_bank_marketing.py](codex/improve_bank_marketing.py) | [improved_model.py](Claude/improved_model.py) | [Antigravity_performance improvement.ipynb](Antigravity/Antigravity_performance%20improvement.ipynb) |
| Leakage Audit | [leakage_analysis_baseline_codex.ipynb](codex/leakage_analysis_baseline_codex.ipynb) | [leakage_analysis.ipynb](Claude/leakage_analysis.ipynb) | [Antigravity_data_leakage.ipynb](Antigravity/Antigravity_data_leakage.ipynb) |
| Interaction Log | [interaction_log.md](codex/interaction_log.md) | [session_log.txt](Claude/session_log.txt) | [interactions_log_Antigravity.md](Antigravity/interactions_log_Antigravity.md) |

## Results at a Glance

| Metric | Codex LR Baseline | Codex RF Improved | Claude LR Baseline | Claude HGBC Improved | Antigravity LR Baseline | Antigravity RF Improved |
|--------|-------------------|-------------------|--------------------|-----------------------|-------------------------|------------------------|
| AUROC | 0.8006 | 0.8063 | 0.8009 | 0.8163 | 0.8010 | 0.8128 |
| PR-AUC | 0.4597 | 0.4905 | 0.4603 | 0.4868 | 0.4601 | 0.4912 |
| F1 (yes) | 0.466 | 0.521 | 0.468 | 0.529 | 0.47 | 0.50 |

All three baselines excluded `duration` and report near-identical LR performance (AUROC ~0.80), confirming valid independent replication. Codex's initial baseline included `duration` (AUROC=0.94) but corrected it in the leakage audit. Improvement strategies differed: Codex and Antigravity chose Random Forest, while Claude used HistGradientBoosting with threshold tuning.

## How to Reproduce

```bash
git clone https://github.com/ben-dom393/Agent_Tooling_Comparison.git
cd Agent_Tooling_Comparison
```

Navigate into any agent folder (`codex/`, `Claude/`, or `Antigravity/`) and run the `.py` scripts directly or open the `.ipynb` notebooks with Jupyter. Requires **Python 3.10+** with `scikit-learn`, `pandas`, `matplotlib`, and `seaborn`.

## Dataset

**UCI Bank Marketing** (`bank-additional-full.csv`) — 41,188 rows, 21 columns, binary target — whether the client subscribed to a term deposit (~11.3% positive class). Each agent folder contains its own copy of the CSV.

---

*Built for UCL MSIN0097 Predictive Analytics 2025-26*
