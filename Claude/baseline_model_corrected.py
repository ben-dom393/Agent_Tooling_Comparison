"""
Baseline Model (CORRECTED) -- Bank Marketing Term-Deposit Prediction
=====================================================================
Target : y  (binary: yes / no)
Approach: Logistic Regression + Random Forest baselines
Metric : AUROC (primary), F1, PR-AUC

Fixes applied vs. the leaked version:
  1. Dropped 'duration' (post-hoc leakage) and 'y' (target in string form)
  2. Removed 'job_yes_rate' (target encoding computed on full dataset)
  3. Moved preprocessing inside Pipeline (was fit_transform on full data)
  4. Split raw data first, then transform via Pipeline
"""

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = pathlib.Path(__file__).parent / "bank-additional-full.csv"
OUT_DIR = DATA_PATH.parent

# =====================================================================
# 1.  LOAD & PREPARE
# =====================================================================
print("=" * 72)
print("1. LOAD & PREPARE DATA")
print("=" * 72)

df = pd.read_csv(DATA_PATH, sep=";")
print(f"Raw shape: {df.shape}")

# --- Encode target -------------------------------------------------------
df["y_binary"] = (df["y"] == "yes").astype(int)

# --- FIX 1: Drop 'duration' (post-hoc leakage) AND 'y' (string target) --
df = df.drop(columns=["duration", "y"])
print("[FIX 1] Dropped 'duration' (post-hoc leakage) and 'y' (string target column).")

# --- FIX 2: Do NOT create job_yes_rate on the full dataset ---------------
#     The leaked version computed: job_yes_rate = df.groupby("job")["y_binary"].mean()
#     This used test-set target values to build a training feature.
#     Removed entirely -- a safe target encoder would need to be inside the
#     Pipeline and fit per fold, which is over-engineering for a baseline.
print("[FIX 2] Removed 'job_yes_rate' (target encoding computed on full data).")

# --- Feature engineering: pdays sentinel --> binary flag ------------------
df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
print(f"[FEATURE] 'was_contacted_before' created "
      f"(1 if pdays!=999: {df['was_contacted_before'].sum():,} rows).\n")

# --- Identify column groups -----------------------------------------------
target = "y_binary"
drop_cols = [target]

cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in df.select_dtypes(include="number").columns.tolist()
            if c != target]

print(f"Numeric features  ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

# Sanity checks: target and leakage columns must not be in features
assert "y" not in cat_cols, "'y' (string target) still in categorical features!"
assert "duration" not in num_cols, "'duration' still in numeric features!"
assert "job_yes_rate" not in num_cols, "'job_yes_rate' still in numeric features!"
print("[CHECK] Confirmed: 'y', 'duration', 'job_yes_rate' are NOT in features.")

X = df.drop(columns=drop_cols)
y = df[target]

# =====================================================================
# 2.  TRAIN / TEST SPLIT  (stratified, 80/20)
#     FIX 3: Split RAW data first, THEN preprocess inside Pipeline.
#     The leaked version did: preprocessor.fit_transform(X) then split.
# =====================================================================
print("\n" + "=" * 72)
print("2. TRAIN / TEST SPLIT")
print("=" * 72)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Train prevalence: {y_train.mean():.4f}  |  Test prevalence: {y_test.mean():.4f}")
print("[FIX 3] Split raw data first. Preprocessing will be fit inside Pipeline on train only.")

# =====================================================================
# 3.  PREPROCESSING (inside Pipeline -- fit on train only)
# =====================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              sparse_output=False, drop="first"), cat_cols),
    ],
    remainder="drop",
)

# =====================================================================
# 4.  DEFINE BASELINE MODELS (wrapped in Pipeline)
# =====================================================================
models = {
    "LogisticRegression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ]),
    "RandomForest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
}

# =====================================================================
# 5.  CROSS-VALIDATION ON TRAINING SET  (5-fold, stratified)
# =====================================================================
print("\n" + "=" * 72)
print("3. 5-FOLD STRATIFIED CROSS-VALIDATION (training set only)")
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
# 6.  HOLDOUT TEST-SET EVALUATION
# =====================================================================
print("\n" + "=" * 72)
print("4. HOLDOUT TEST-SET EVALUATION")
print("=" * 72)

holdout_results = {}
fitted_models = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    holdout_results[name] = {"AUROC": auroc, "PR-AUC": pr_auc, "F1": f1}

    print(f"\n  {name}:")
    print(f"    AUROC  = {auroc:.4f}")
    print(f"    PR-AUC = {pr_auc:.4f}")
    print(f"    F1     = {f1:.4f}")
    print(f"\n    Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"], digits=4))

# =====================================================================
# 7.  SUMMARY TABLE
# =====================================================================
print("\n" + "=" * 72)
print("5. COMBINED RESULTS SUMMARY")
print("=" * 72)

rows = []
for name in models:
    row = {"Model": name}
    for metric in cv_scoring:
        m, s = cv_results[name][metric]
        row[f"CV {metric} (mean+/-std)"] = f"{m:.4f}+/-{s:.4f}"
    for metric, val in holdout_results[name].items():
        row[f"Holdout {metric}"] = f"{val:.4f}"
    rows.append(row)

results_df = pd.DataFrame(rows)
print(results_df.to_string(index=False))
results_df.to_csv(OUT_DIR / "baseline_corrected_results.csv", index=False)

# =====================================================================
# 8.  FIGURES
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for name, pipe in fitted_models.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
ax.set_title("ROC Curve -- Holdout Set (Corrected)")
ax.legend(loc="lower right")

ax = axes[1]
for name, pipe in fitted_models.items():
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
prevalence = y_test.mean()
ax.axhline(y=prevalence, color="k", linestyle="--", lw=1,
           label=f"No-skill ({prevalence:.3f})")
ax.set_title("Precision-Recall Curve -- Holdout Set (Corrected)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_baseline_corrected_roc_pr.png", dpi=150)
plt.close()
print("\nFigure saved: fig_baseline_corrected_roc_pr.png")

# =====================================================================
# 9.  FEATURE IMPORTANCE (Logistic Regression coefficients)
# =====================================================================
print("\n" + "=" * 72)
print("6. TOP-20 LOGISTIC REGRESSION COEFFICIENTS (magnitude)")
print("=" * 72)

lr_pipe = fitted_models["LogisticRegression"]
ohe_feature_names = (
    lr_pipe.named_steps["pre"]
    .named_transformers_["cat"]
    .get_feature_names_out(cat_cols)
    .tolist()
)
all_features = num_cols + ohe_feature_names
coefs = lr_pipe.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({"feature": all_features, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

print(coef_df.head(20).to_string(index=False))

# =====================================================================
# 10.  VALIDITY CHECKS (corrected)
# =====================================================================
print("\n" + "=" * 72)
print("7. VALIDITY CHECKS (corrected)")
print("=" * 72)

print("  [OK] 'duration' dropped before any modelling (post-hoc leakage)")
print("  [OK] 'y' (string target) dropped before feature construction")
print("  [OK] 'job_yes_rate' removed (was target encoding on full data)")
print("  [OK] Preprocessing inside Pipeline -- fit on train only per fold")
print("  [OK] Stratified split preserves class ratio:")
print(f"       Train prevalence = {y_train.mean():.4f}")
print(f"       Test prevalence  = {y_test.mean():.4f}")
print("  [OK] Cross-validation is stratified with shuffle + fixed seed")

assert len(set(X_train.index) & set(X_test.index)) == 0
print("  [OK] Zero index overlap between train and test sets")
assert "y" not in X_train.columns
assert "duration" not in X_train.columns
assert "job_yes_rate" not in X_train.columns
print("  [OK] Leakage features confirmed absent from train/test data")

print("\n" + "=" * 72)
print("CORRECTED BASELINE PIPELINE COMPLETE")
print("=" * 72)
