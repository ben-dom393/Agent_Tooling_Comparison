"""
Improved Model -- Bank Marketing Term-Deposit Prediction
=========================================================
Improvements over baseline (Logistic Regression, AUROC=0.80):
  1. Gradient Boosting (HistGradientBoostingClassifier) for non-linearity
  2. Better feature engineering (ordinal education, pdays cleanup)
  3. CV-based threshold tuning for F1

Fair comparison: same train/test split, same metrics, same random seed.
"""

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = pathlib.Path(__file__).parent / "bank-additional-full.csv"
OUT_DIR = DATA_PATH.parent

# =====================================================================
# 1.  LOAD & SHARED PREPROCESSING  (identical to baseline)
# =====================================================================
print("=" * 72)
print("1. LOAD & PREPARE DATA")
print("=" * 72)

df = pd.read_csv(DATA_PATH, sep=";")
print(f"Raw shape: {df.shape}")

df["y_binary"] = (df["y"] == "yes").astype(int)
df = df.drop(columns=["duration", "y"])
print("[LEAKAGE] Dropped 'duration'.")

# --- Feature engineering (improvement #2) --------------------------------

# 2a. Binary contact flag (same as baseline)
df["was_contacted_before"] = (df["pdays"] != 999).astype(int)

# 2b. Clean pdays: replace 999 sentinel with -1 for tree-based models
#     This separates "never contacted" from actual day counts.
#     (LR baseline used raw 999 which is misleading in a linear model.)
df["pdays_clean"] = df["pdays"].replace(999, -1)

# 2c. Ordinal encoding for education (natural order exists)
edu_order = [
    "illiterate", "basic.4y", "basic.6y", "basic.9y",
    "high.school", "professional.course", "university.degree", "unknown",
]
df["education_ordinal"] = df["education"].map(
    {v: i for i, v in enumerate(edu_order)}
)

# 2d. Campaign intensity flag: contacted more than 3 times this campaign
df["high_campaign"] = (df["campaign"] > 3).astype(int)

# 2e. Total previous contacts (already exists as 'previous', just confirm)
print(f"[FEATURE] Created: pdays_clean, education_ordinal, high_campaign, was_contacted_before")

# --- Define feature groups -----------------------------------------------
target = "y_binary"

# For the BASELINE (Logistic Regression) -- use same features as original
cat_cols_baseline = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
num_cols_baseline = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed", "was_contacted_before",
]

# For the IMPROVED model -- use new features, drop raw pdays
cat_cols_improved = [
    "job", "marital", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]  # education handled ordinally
num_cols_improved = [
    "age", "campaign", "pdays_clean", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed", "was_contacted_before",
    "education_ordinal", "high_campaign",
]

X = df.drop(columns=[target])
y = df[target]

# =====================================================================
# 2.  IDENTICAL TRAIN/TEST SPLIT  (same seed = same split as baseline)
# =====================================================================
print("\n" + "=" * 72)
print("2. TRAIN / TEST SPLIT (same as baseline)")
print("=" * 72)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Train prevalence: {y_train.mean():.4f}  |  Test prevalence: {y_test.mean():.4f}")

# =====================================================================
# 3.  DEFINE MODELS
# =====================================================================

# 3a. BASELINE: Logistic Regression (exact replica)
baseline_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_baseline),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              sparse_output=False, drop="first"), cat_cols_baseline),
    ],
    remainder="drop",
)

baseline_pipe = Pipeline([
    ("pre", baseline_preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_STATE,
    )),
])

# 3b. IMPROVED: HistGradientBoosting with better features
improved_preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols_improved),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), cat_cols_improved),
    ],
    remainder="drop",
)

improved_pipe = Pipeline([
    ("pre", improved_preprocessor),
    ("clf", HistGradientBoostingClassifier(
        max_iter=300,           # enough trees; early stopping prevents overfit
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
        categorical_features=[
            len(num_cols_improved) + i for i in range(len(cat_cols_improved))
        ],
    )),
])

models = {
    "LR_baseline": baseline_pipe,
    "HGBC_improved": improved_pipe,
}

# =====================================================================
# 4.  5-FOLD STRATIFIED CROSS-VALIDATION  (training set only)
# =====================================================================
print("\n" + "=" * 72)
print("3. 5-FOLD STRATIFIED CV (training set only)")
print("=" * 72)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scoring = ["roc_auc", "f1", "average_precision"]

cv_results = {}
for name, pipe in models.items():
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv, scoring=cv_scoring,
        return_train_score=False, n_jobs=-1,
    )
    cv_results[name] = {
        metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
        for metric in cv_scoring
    }
    print(f"\n  {name}:")
    for metric in cv_scoring:
        m, s = cv_results[name][metric]
        print(f"    {metric:25s}  {m:.4f} +/- {s:.4f}")

# =====================================================================
# 5.  CV-BASED THRESHOLD TUNING  (improvement #3)
#     Find the threshold that maximises F1 on inner CV folds.
#     This avoids using the test set for threshold selection.
# =====================================================================
print("\n" + "=" * 72)
print("4. CV-BASED THRESHOLD TUNING (improved model only)")
print("=" * 72)


def find_best_threshold_cv(pipe, X_tr, y_tr, cv_splitter):
    """Find the probability threshold maximising F1 across CV folds."""
    thresholds = np.arange(0.20, 0.65, 0.01)
    fold_best_thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_tr, y_tr)):
        X_fold_train = X_tr.iloc[train_idx]
        y_fold_train = y_tr.iloc[train_idx]
        X_fold_val = X_tr.iloc[val_idx]
        y_fold_val = y_tr.iloc[val_idx]

        pipe_clone = pipe.__class__(pipe.steps)
        # Deep-clone the pipeline for each fold
        from sklearn.base import clone
        pipe_clone = clone(pipe)
        pipe_clone.fit(X_fold_train, y_fold_train)
        y_prob_val = pipe_clone.predict_proba(X_fold_val)[:, 1]

        best_f1 = 0
        best_t = 0.5
        for t in thresholds:
            f1_val = f1_score(y_fold_val, (y_prob_val >= t).astype(int))
            if f1_val > best_f1:
                best_f1 = f1_val
                best_t = t
        fold_best_thresholds.append(best_t)
        print(f"    Fold {fold_idx + 1}: best threshold = {best_t:.2f}, F1 = {best_f1:.4f}")

    avg_threshold = np.mean(fold_best_thresholds)
    print(f"    --> Average optimal threshold: {avg_threshold:.3f}")
    return avg_threshold


best_threshold = find_best_threshold_cv(improved_pipe, X_train, y_train, cv)

# =====================================================================
# 6.  HOLDOUT TEST-SET EVALUATION
# =====================================================================
print("\n" + "=" * 72)
print("5. HOLDOUT TEST-SET EVALUATION")
print("=" * 72)

holdout_results = {}
fitted_models = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Default threshold for baseline, tuned threshold for improved
    if name == "HGBC_improved":
        threshold = best_threshold
    else:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    holdout_results[name] = {
        "AUROC": auroc, "PR-AUC": pr_auc, "F1": f1, "threshold": threshold,
    }

    print(f"\n  {name} (threshold={threshold:.3f}):")
    print(f"    AUROC  = {auroc:.4f}")
    print(f"    PR-AUC = {pr_auc:.4f}")
    print(f"    F1     = {f1:.4f}")
    print(f"\n    Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"], digits=4))

# =====================================================================
# 7.  COMBINED RESULTS SUMMARY TABLE
# =====================================================================
print("\n" + "=" * 72)
print("6. COMBINED RESULTS: BASELINE vs IMPROVED")
print("=" * 72)

rows = []
for name in models:
    row = {"Model": name}
    for metric in cv_scoring:
        m, s = cv_results[name][metric]
        row[f"CV_{metric}"] = f"{m:.4f}+/-{s:.4f}"
    for metric, val in holdout_results[name].items():
        if metric == "threshold":
            row["Threshold"] = f"{val:.3f}"
        else:
            row[f"Holdout_{metric}"] = f"{val:.4f}"
    rows.append(row)

results_df = pd.DataFrame(rows)
print(results_df.to_string(index=False))
results_df.to_csv(OUT_DIR / "improved_results_summary.csv", index=False)

# --- Compute deltas --------------------------------------------------------
print("\n  Performance delta (improved - baseline):")
for metric in ["AUROC", "PR-AUC", "F1"]:
    base = holdout_results["LR_baseline"][metric]
    impr = holdout_results["HGBC_improved"][metric]
    delta = impr - base
    pct = delta / base * 100
    print(f"    {metric:8s}:  {base:.4f} -> {impr:.4f}  (delta={delta:+.4f}, {pct:+.1f}%)")

# =====================================================================
# 8.  FIGURES
# =====================================================================

# 8a. ROC + PR curves: baseline vs improved
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for name, pipe in fitted_models.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
ax.set_title("ROC Curve -- Baseline vs Improved")
ax.legend(loc="lower right")

ax = axes[1]
for name, pipe in fitted_models.items():
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
prevalence = y_test.mean()
ax.axhline(y=prevalence, color="k", linestyle="--", lw=1,
           label=f"No-skill ({prevalence:.3f})")
ax.set_title("Precision-Recall Curve -- Baseline vs Improved")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_improved_roc_pr.png", dpi=150)
plt.close()
print("\nFigure saved: fig_improved_roc_pr.png")

# 8b. Feature importance via permutation importance (more robust)
from sklearn.inspection import permutation_importance

hgbc_model = fitted_models["HGBC_improved"]
perm_result = permutation_importance(
    hgbc_model, X_test, y_test, n_repeats=10,
    random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
)

all_feat_names = X_test.columns.tolist()
imp_df = pd.DataFrame({
    "feature": all_feat_names,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
})
imp_df = imp_df.sort_values("importance_mean", ascending=False)
print("\nTop-15 features by permutation importance (AUROC):")
print(imp_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 7))
top15 = imp_df.head(15).sort_values("importance_mean")
ax.barh(top15["feature"], top15["importance_mean"],
        xerr=top15["importance_std"], color="#4878cf", edgecolor="black")
ax.set_xlabel("Permutation Importance (decrease in AUROC)")
ax.set_title("Top-15 Features -- HistGradientBoosting (Improved Model)")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_improved_feature_importance.png", dpi=150)
plt.close()
print("Figure saved: fig_improved_feature_importance.png")

# 8c. Summary bar chart: baseline vs improved on all 3 metrics
fig, ax = plt.subplots(figsize=(8, 5))
metrics_to_plot = ["AUROC", "PR-AUC", "F1"]
x = np.arange(len(metrics_to_plot))
width = 0.35
base_vals = [holdout_results["LR_baseline"][m] for m in metrics_to_plot]
impr_vals = [holdout_results["HGBC_improved"][m] for m in metrics_to_plot]

bars1 = ax.bar(x - width / 2, base_vals, width, label="LR Baseline",
               color="#4878cf", edgecolor="black")
bars2 = ax.bar(x + width / 2, impr_vals, width, label="HGBC Improved",
               color="#e1812c", edgecolor="black")

ax.set_ylabel("Score")
ax.set_title("Holdout Metrics: Baseline vs Improved")
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.legend()
ax.set_ylim(0, 1)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_baseline_vs_improved.png", dpi=150)
plt.close()
print("Figure saved: fig_baseline_vs_improved.png")

# =====================================================================
# 9.  LEAKAGE & RISK CHECKS
# =====================================================================
print("\n" + "=" * 72)
print("7. LEAKAGE & RISK CHECKS")
print("=" * 72)

print("  [OK] 'duration' excluded from both models")
print("  [OK] Same train/test split (same random_state=42)")
print("  [OK] Preprocessing fitted on train only (inside Pipeline)")
print("  [OK] Threshold tuned on inner CV folds -- test set never seen")
print("  [OK] No target-informed feature engineering")
print("  [OK] Early stopping uses a held-out validation fraction from train only")
print(f"  [OK] Stratified split: train prev={y_train.mean():.4f}, test prev={y_test.mean():.4f}")

assert len(set(X_train.index) & set(X_test.index)) == 0
print("  [OK] Zero index overlap between train and test")

# --- Overfit check: compare CV vs holdout --------------------------------
print("\n  Overfit check (CV AUROC vs Holdout AUROC):")
for name in models:
    cv_auroc = cv_results[name]["roc_auc"][0]
    ho_auroc = holdout_results[name]["AUROC"]
    gap = ho_auroc - cv_auroc
    print(f"    {name:20s}  CV={cv_auroc:.4f}  Holdout={ho_auroc:.4f}  gap={gap:+.4f}")

print("\n" + "=" * 72)
print("IMPROVED PIPELINE COMPLETE")
print("=" * 72)
