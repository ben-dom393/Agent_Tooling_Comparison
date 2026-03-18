"""
Exploratory Data Analysis -- Bank Marketing Dataset
=====================================================
Target: y (binary: yes / no) -- did the client subscribe to a term deposit?
Data:   bank-additional-full.csv (UCI Bank Marketing, semicolon-delimited)
"""

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05)

DATA_PATH = pathlib.Path(__file__).parent / "bank-additional-full.csv"
RANDOM_STATE = 42

# ── 1. Load & inspect ───────────────────────────────────────────────────────
print("=" * 72)
print("1. LOAD & INSPECT")
print("=" * 72)

df = pd.read_csv(DATA_PATH, sep=";")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
print(df.dtypes.to_string())
print("\nFirst 5 rows:")
print(df.head().to_string())

# ── 2. Missing / special values ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. MISSING & SPECIAL VALUES")
print("=" * 72)

true_nan = df.isna().sum()
print("True NaN per column:")
print(true_nan[true_nan > 0].to_string() if true_nan.sum() > 0 else "  None")

cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

# "unknown" acts as missing in categoricals
unknown_counts = {}
for c in cat_cols:
    n = (df[c] == "unknown").sum()
    if n > 0:
        unknown_counts[c] = n
unknown_df = pd.DataFrame.from_dict(
    unknown_counts, orient="index", columns=["unknown_count"]
)
unknown_df["pct"] = (unknown_df["unknown_count"] / len(df) * 100).round(2)
print('\n"unknown" entries per categorical column:')
print(unknown_df.sort_values("pct", ascending=False).to_string())

# ── 3. Summary statistics ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. SUMMARY STATISTICS")
print("=" * 72)

print("\nNumeric features:")
print(df[num_cols].describe().T.to_string())

print("\nCategorical features -- unique values:")
for c in cat_cols:
    vals = df[c].value_counts()
    print(f"\n  {c} ({vals.shape[0]} levels):")
    print("    " + vals.to_string().replace("\n", "\n    "))

# ── 4. Target distribution ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("4. TARGET DISTRIBUTION")
print("=" * 72)

target_counts = df["y"].value_counts()
target_pct = df["y"].value_counts(normalize=True) * 100
target_summary = pd.DataFrame({"count": target_counts, "pct": target_pct.round(2)})
print(target_summary.to_string())

fig, ax = plt.subplots(figsize=(5, 4))
target_counts.plot.bar(ax=ax, color=["#4878cf", "#e1812c"], edgecolor="black")
ax.set_title("Target distribution (y)")
ax.set_ylabel("Count")
for i, v in enumerate(target_counts):
    ax.text(i, v + 200, f"{v:,}\n({target_pct.iloc[i]:.1f}%)", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(DATA_PATH.parent / "fig_target_distribution.png", dpi=150)
plt.close()

# ── 5. Univariate analysis ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("5. UNIVARIATE DISTRIBUTIONS")
print("=" * 72)

# 5a. Numeric histograms
num_features = [c for c in num_cols if c != "y"]
n_num = len(num_features)
ncols_fig = 3
nrows_fig = int(np.ceil(n_num / ncols_fig))

fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(5 * ncols_fig, 4 * nrows_fig))
axes = axes.flatten()
for i, col in enumerate(num_features):
    ax = axes[i]
    ax.hist(df[col], bins=50, edgecolor="black", alpha=0.7, color="#4878cf")
    ax.set_title(col)
    ax.set_ylabel("Frequency")
    skew_val = df[col].skew()
    ax.text(
        0.95, 0.92, f"skew={skew_val:.2f}",
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
    )
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Numeric feature distributions", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(DATA_PATH.parent / "fig_numeric_distributions.png", dpi=150)
plt.close()

# 5b. Categorical bar charts
n_cat = len(cat_cols) - 1  # exclude target y
ncols_cat = 3
nrows_cat = int(np.ceil(n_cat / ncols_cat))
cat_features = [c for c in cat_cols if c != "y"]

fig, axes = plt.subplots(nrows_cat, ncols_cat, figsize=(5 * ncols_cat, 4 * nrows_cat))
axes = axes.flatten()
for i, col in enumerate(cat_features):
    ax = axes[i]
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, ax=ax, color="#4878cf", edgecolor="black")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=45)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Categorical feature distributions", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(DATA_PATH.parent / "fig_categorical_distributions.png", dpi=150)
plt.close()

# ── 6. Bivariate analysis ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("6. BIVARIATE ANALYSIS -- predictors vs target")
print("=" * 72)

# 6a. Subscription rate by each categorical predictor
print("\nSubscription rate (% yes) by categorical feature:")
sub_rate_records = []
for col in cat_features:
    ct = pd.crosstab(df[col], df["y"])
    ct["total"] = ct.sum(axis=1)
    ct["sub_rate_pct"] = (ct["yes"] / ct["total"] * 100).round(2)
    print(f"\n  {col}:")
    print("    " + ct[["total", "sub_rate_pct"]].sort_values("sub_rate_pct", ascending=False).to_string().replace("\n", "\n    "))
    for level, row in ct.iterrows():
        sub_rate_records.append(
            {"feature": col, "level": level, "total": row["total"], "sub_rate_pct": row["sub_rate_pct"]}
        )
sub_rate_df = pd.DataFrame(sub_rate_records)

# Plot subscription rates for each categorical feature
fig, axes = plt.subplots(nrows_cat, ncols_cat, figsize=(6 * ncols_cat, 4 * nrows_cat))
axes = axes.flatten()
for i, col in enumerate(cat_features):
    ax = axes[i]
    sub = sub_rate_df[sub_rate_df["feature"] == col].sort_values("sub_rate_pct", ascending=True)
    ax.barh(sub["level"], sub["sub_rate_pct"], color="#e1812c", edgecolor="black")
    ax.set_xlabel("Subscription rate (%)")
    ax.set_title(col)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Subscription rate by categorical feature", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(DATA_PATH.parent / "fig_sub_rate_by_categorical.png", dpi=150)
plt.close()

# 6b. Box plots of numerics by target
fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(5 * ncols_fig, 4 * nrows_fig))
axes = axes.flatten()
for i, col in enumerate(num_features):
    ax = axes[i]
    sns.boxplot(data=df, x="y", y=col, ax=ax, palette=["#4878cf", "#e1812c"])
    ax.set_title(col)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Numeric features by target (y)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(DATA_PATH.parent / "fig_boxplots_by_target.png", dpi=150)
plt.close()

# 6c. Correlation heatmap (numeric features only)
corr = df[num_cols].copy()
corr["y_binary"] = (df["y"] == "yes").astype(int)
corr_matrix = corr.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
)
ax.set_title("Correlation matrix (numeric features + target)")
plt.tight_layout()
plt.savefig(DATA_PATH.parent / "fig_correlation_heatmap.png", dpi=150)
plt.close()

print("\nCorrelation with target (y_binary):")
target_corr = corr_matrix["y_binary"].drop("y_binary").sort_values(key=abs, ascending=False)
print(target_corr.to_string())

# ── 7. Leakage & special-value checks ───────────────────────────────────────
print("\n" + "=" * 72)
print("7. DATA LEAKAGE & SPECIAL-VALUE CHECKS")
print("=" * 72)

# 7a. duration -- known only after the call ends
dur_corr = corr_matrix.loc["duration", "y_binary"]
print(f"\n[LEAKAGE] duration correlation with target: {dur_corr:.3f}")
print("  'duration' is the call length in seconds. It is only known AFTER the")
print("  call, so using it in a predictive model constitutes data leakage.")
print("  It should be EXCLUDED from production models.")

# 7b. pdays sentinel value 999
pdays_999 = (df["pdays"] == 999).sum()
print(f"\n[SPECIAL] pdays == 999 (not previously contacted): {pdays_999:,} "
      f"({pdays_999 / len(df) * 100:.1f}%)")

# 7c. Consistency: pdays=999 ↔ previous=0 ↔ poutcome=nonexistent
mask_999 = df["pdays"] == 999
inconsistent = df[mask_999 & ((df["previous"] != 0) | (df["poutcome"] != "nonexistent"))]
print(f"  Inconsistent rows (pdays=999 but previous!=0 or poutcome!=nonexistent): {len(inconsistent)}")

mask_not999 = df["pdays"] != 999
inconsistent2 = df[mask_not999 & ((df["previous"] == 0) | (df["poutcome"] == "nonexistent"))]
print(f"  Inconsistent rows (pdays!=999 but previous==0 or poutcome==nonexistent): {len(inconsistent2)}")

# ── 8. Outlier / distributional flags ────────────────────────────────────────
print("\n" + "=" * 72)
print("8. OUTLIER & DISTRIBUTIONAL FLAGS")
print("=" * 72)

skewness = df[num_features].skew().sort_values(key=abs, ascending=False)
print("\nSkewness of numeric features (|skew| > 1 flagged):")
for feat, sk in skewness.items():
    flag = " << HIGHLY SKEWED" if abs(sk) > 1 else ""
    print(f"  {feat:25s} {sk:+.3f}{flag}")

# Outlier detection via IQR
print("\nIQR-based outlier counts:")
outlier_info = []
for col in num_features:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_info.append({"feature": col, "q1": q1, "q3": q3, "iqr": iqr,
                         "lower": lower, "upper": upper, "n_outliers": n_out,
                         "pct_outliers": round(n_out / len(df) * 100, 2)})
    print(f"  {col:25s} {n_out:6,} outliers ({n_out / len(df) * 100:.1f}%)")
outlier_df = pd.DataFrame(outlier_info)

# ── 9. Summary table (all features) ─────────────────────────────────────────
print("\n" + "=" * 72)
print("9. COMPREHENSIVE FEATURE SUMMARY TABLE")
print("=" * 72)

summary_rows = []
y_binary = (df["y"] == "yes").astype(int)

for col in df.columns:
    if col == "y":
        continue
    row = {"feature": col}
    if col in num_cols:
        row["type"] = "numeric"
        row["n_unique"] = df[col].nunique()
        row["missing_or_unknown"] = int(df[col].isna().sum())
        row["mean"] = round(df[col].mean(), 3)
        row["std"] = round(df[col].std(), 3)
        row["median"] = round(df[col].median(), 3)
        row["skewness"] = round(df[col].skew(), 3)
        # point-biserial correlation with target
        row["corr_with_y"] = round(df[col].corr(y_binary), 3)
        out_row = outlier_df[outlier_df["feature"] == col]
        row["pct_outliers"] = float(out_row["pct_outliers"].values[0]) if len(out_row) else 0
    else:
        row["type"] = "categorical"
        row["n_unique"] = df[col].nunique()
        unk = int((df[col] == "unknown").sum())
        row["missing_or_unknown"] = unk
        row["mean"] = np.nan
        row["std"] = np.nan
        row["median"] = np.nan
        row["skewness"] = np.nan
        row["corr_with_y"] = np.nan
        row["pct_outliers"] = np.nan
    summary_rows.append(row)

summary_table = pd.DataFrame(summary_rows)
summary_table = summary_table[
    ["feature", "type", "n_unique", "missing_or_unknown",
     "mean", "std", "median", "skewness", "corr_with_y", "pct_outliers"]
]
print(summary_table.to_string(index=False))

# Save summary table as CSV
summary_table.to_csv(DATA_PATH.parent / "eda_summary_table.csv", index=False)

# ── 10. Summary figure -- feature importance proxy (|corr| with target) ──────
fig, ax = plt.subplots(figsize=(8, 6))
num_summary = summary_table[summary_table["type"] == "numeric"].copy()
num_summary["abs_corr"] = num_summary["corr_with_y"].abs()
num_summary = num_summary.sort_values("abs_corr", ascending=True)
colors = ["#d62728" if v < 0 else "#2ca02c" for v in num_summary["corr_with_y"]]
ax.barh(num_summary["feature"], num_summary["corr_with_y"], color=colors, edgecolor="black")
ax.set_xlabel("Pearson correlation with target (y)")
ax.set_title("Numeric features: correlation with subscription (y=yes)")
ax.axvline(0, color="black", linewidth=0.8)
for i, (feat, val) in enumerate(zip(num_summary["feature"], num_summary["corr_with_y"])):
    ax.text(val + 0.01 * np.sign(val), i, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(DATA_PATH.parent / "fig_feature_correlation_with_target.png", dpi=150)
plt.close()

print("\n" + "=" * 72)
print("EDA COMPLETE. Figures saved to the working directory.")
print("=" * 72)
