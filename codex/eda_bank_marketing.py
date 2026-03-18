from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def make_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def infer_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]
    return numeric_cols, categorical_cols


def basic_structure_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "n_unique": int(series.nunique(dropna=True)),
                "sample_values": ", ".join(map(str, series.dropna().astype(str).unique()[:5])),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing", "n_unique"], ascending=[False, False])


def unknown_token_table(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        unknown_mask = df[col].astype(str).str.lower().eq("unknown")
        rows.append(
            {
                "column": col,
                "unknown_count": int(unknown_mask.sum()),
                "unknown_pct": float(unknown_mask.mean() * 100),
            }
        )
    return pd.DataFrame(rows).sort_values("unknown_pct", ascending=False)


def numeric_summary_table(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    summary = df[numeric_cols].describe().T
    summary["missing"] = df[numeric_cols].isna().sum()
    summary["skew"] = df[numeric_cols].skew(numeric_only=True)
    return summary.reset_index(names="column")


def target_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = df[target].value_counts(dropna=False).rename_axis(target).reset_index(name="count")
    counts["pct"] = counts["count"] / len(df) * 100
    return counts


def numeric_target_relationships(
    df: pd.DataFrame, numeric_cols: list[str], target: str
) -> pd.DataFrame:
    target_binary = df[target].map({"no": 0, "yes": 1})
    rows = []
    for col in numeric_cols:
        x = df[col]
        valid = x.notna() & target_binary.notna()
        x_valid = x[valid]
        y_valid = target_binary[valid]
        if y_valid.nunique() < 2:
            continue

        group0 = x_valid[y_valid == 0]
        group1 = x_valid[y_valid == 1]
        mean0 = group0.mean()
        mean1 = group1.mean()
        pooled_sd = np.sqrt((group0.var(ddof=1) + group1.var(ddof=1)) / 2)
        effect = 0.0 if pooled_sd == 0 or np.isnan(pooled_sd) else (mean1 - mean0) / pooled_sd
        corr = np.corrcoef(x_valid, y_valid)[0, 1]
        try:
            _, p_value = stats.mannwhitneyu(group0, group1, alternative="two-sided")
        except ValueError:
            p_value = np.nan

        rows.append(
            {
                "feature": col,
                "tool": "numeric_vs_binary_target",
                "task": "screening",
                "mean_target_no": mean0,
                "mean_target_yes": mean1,
                "std_mean_diff": effect,
                "pearson_with_target": corr,
                "mannwhitney_pvalue": p_value,
                "zero_share": float((x_valid == 0).mean() * 100),
            }
        )
    return pd.DataFrame(rows).sort_values("std_mean_diff", key=lambda s: s.abs(), ascending=False)


def cramers_v(confusion: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan


def categorical_target_relationships(
    df: pd.DataFrame, categorical_cols: list[str], target: str
) -> pd.DataFrame:
    rows = []
    overall_positive_rate = (df[target] == "yes").mean()
    for col in categorical_cols:
        ctab = pd.crosstab(df[col], df[target])
        if ctab.shape[1] < 2:
            continue
        rate_by_level = df.groupby(col, dropna=False)[target].apply(lambda s: (s == "yes").mean())
        count_by_level = df.groupby(col, dropna=False)[target].size()
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(ctab)
        except ValueError:
            chi2, p_value = np.nan, np.nan

        strongest_level = (rate_by_level - overall_positive_rate).abs().sort_values(ascending=False).index[0]
        rows.append(
            {
                "feature": col,
                "tool": "categorical_vs_binary_target",
                "task": "screening",
                "n_levels": int(df[col].nunique(dropna=False)),
                "largest_level_share_pct": float(count_by_level.max() / len(df) * 100),
                "top_lift_level": str(strongest_level),
                "top_lift_level_rate": float(rate_by_level.loc[strongest_level] * 100),
                "overall_target_rate": float(overall_positive_rate * 100),
                "cramers_v": cramers_v(ctab),
                "chi2_pvalue": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)


def leakage_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "feature": "duration",
            "risk_level": "high",
            "reason": "Call duration is usually known only during/after the contact, so it may not exist at scoring time.",
        },
        {
            "feature": "month/day_of_week",
            "risk_level": "medium",
            "reason": "Calendar fields can encode campaign scheduling rather than customer propensity; safe only if available at prediction time.",
        },
        {
            "feature": "pdays/previous/poutcome",
            "risk_level": "medium",
            "reason": "Prior-contact history is useful but must be defined using information strictly available before the prediction timestamp.",
        },
    ]
    available = pd.DataFrame(rows)
    return available[available["feature"].apply(lambda x: any(part in df.columns for part in x.split("/")))]


def plot_target_distribution(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = df[target].value_counts().index
    sns.countplot(data=df, x=target, order=order, ax=ax, palette="Set2", hue=target, dodge=False, legend=False)
    ax.set_title("Target Distribution")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    total = len(df)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height / total:.1%}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)


def plot_missing_and_unknown(structure: pd.DataFrame, unknowns: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    top_missing = structure.sort_values("missing_pct", ascending=False).head(10)
    sns.barplot(data=top_missing, y="column", x="missing_pct", ax=axes[0], color="#d95f02")
    axes[0].set_title("Top Missingness Rates")
    axes[0].set_xlabel("Missing (%)")
    axes[0].set_ylabel("")

    top_unknown = unknowns.sort_values("unknown_pct", ascending=False).head(10)
    sns.barplot(data=top_unknown, y="column", x="unknown_pct", ax=axes[1], color="#1b9e77")
    axes[1].set_title("Encoded 'unknown' Rates")
    axes[1].set_xlabel("Unknown token (%)")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(output_dir / "missing_and_unknown_summary.png", dpi=160)
    plt.close(fig)


def plot_numeric_vs_target(
    df: pd.DataFrame, numeric_rel: pd.DataFrame, target: str, output_dir: Path, top_n: int = 6
) -> None:
    top_features = numeric_rel["feature"].head(top_n).tolist()
    if not top_features:
        return
    ncols = 2
    nrows = int(np.ceil(len(top_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, feature in zip(axes, top_features):
        sns.boxplot(data=df, x=target, y=feature, ax=ax, palette="Set2", hue=target, dodge=False, legend=False)
        ax.set_title(f"{feature} vs {target}")
    for ax in axes[len(top_features) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_vs_target.png", dpi=160)
    plt.close(fig)


def plot_target_rate_by_categorical(
    df: pd.DataFrame, categorical_rel: pd.DataFrame, target: str, output_dir: Path, top_n: int = 4
) -> None:
    top_features = categorical_rel["feature"].head(top_n).tolist()
    if not top_features:
        return
    fig, axes = plt.subplots(len(top_features), 1, figsize=(10, 4 * len(top_features)))
    axes = np.atleast_1d(axes)
    for ax, feature in zip(axes, top_features):
        summary = (
            df.groupby(feature, dropna=False)[target]
            .apply(lambda s: (s == "yes").mean() * 100)
            .sort_values(ascending=False)
            .reset_index(name="target_rate_pct")
        )
        sns.barplot(data=summary, x="target_rate_pct", y=feature, ax=ax, color="#7570b3")
        ax.set_title(f"Positive Rate by {feature}")
        ax.set_xlabel("Target rate (%)")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "categorical_target_rates.png", dpi=160)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], output_dir: Path) -> None:
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def build_issue_table(
    structure: pd.DataFrame,
    unknowns: pd.DataFrame,
    numeric_rel: pd.DataFrame,
    categorical_rel: pd.DataFrame,
) -> pd.DataFrame:
    issues = []

    for _, row in structure.query("missing_pct > 0").iterrows():
        issues.append(
            {
                "feature": row["column"],
                "issue_type": "missing_values",
                "severity": "medium",
                "evidence": f"{row['missing_pct']:.2f}% explicit missing values",
            }
        )

    for _, row in unknowns.query("unknown_pct > 0").iterrows():
        issues.append(
            {
                "feature": row["column"],
                "issue_type": "encoded_missing",
                "severity": "medium" if row["unknown_pct"] >= 5 else "low",
                "evidence": f"{row['unknown_pct']:.2f}% encoded as 'unknown'",
            }
        )

    for _, row in numeric_rel.query("zero_share >= 50").iterrows():
        issues.append(
            {
                "feature": row["feature"],
                "issue_type": "zero_inflation",
                "severity": "medium",
                "evidence": f"{row['zero_share']:.2f}% zeros",
            }
        )

    for _, row in categorical_rel.query("largest_level_share_pct >= 70").iterrows():
        issues.append(
            {
                "feature": row["feature"],
                "issue_type": "dominant_category",
                "severity": "low",
                "evidence": f"Largest level covers {row['largest_level_share_pct']:.2f}% of rows",
            }
        )

    return pd.DataFrame(issues)


def summarise_findings(
    target_dist: pd.DataFrame,
    unknowns: pd.DataFrame,
    numeric_rel: pd.DataFrame,
    categorical_rel: pd.DataFrame,
) -> list[str]:
    findings = []

    positive_rate = target_dist.loc[target_dist.iloc[:, 0] == "yes", "pct"]
    if not positive_rate.empty:
        findings.append(
            f"Target positive rate is {positive_rate.iloc[0]:.2f}%, so the classification problem is materially imbalanced."
        )

    top_unknown = unknowns.sort_values("unknown_pct", ascending=False).head(3)
    unknown_text = ", ".join(
        f"{row.column} ({row.unknown_pct:.1f}%)" for row in top_unknown.itertuples() if row.unknown_pct > 0
    )
    if unknown_text:
        findings.append(f"Several categorical fields use 'unknown' as encoded missingness, especially {unknown_text}.")

    if not numeric_rel.empty:
        top_num = numeric_rel.iloc[0]
        findings.append(
            f"Among numeric predictors, {top_num['feature']} shows the largest standardized separation between target classes "
            f"({top_num['std_mean_diff']:.2f})."
        )

    if not categorical_rel.empty:
        top_cat = categorical_rel.iloc[0]
        findings.append(
            f"Among categorical predictors, {top_cat['feature']} has the strongest association with the target by Cramer's V "
            f"({top_cat['cramers_v']:.2f})."
        )

    return findings


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser(description="Practical EDA for predictive modelling.")
    parser.add_argument("--data", default="bank-additional-full.csv", help="Path to the input CSV.")
    parser.add_argument("--target", default="y", help="Target column.")
    parser.add_argument("--sep", default=";", help="CSV delimiter.")
    parser.add_argument("--output-dir", default="outputs/eda", help="Directory for plots and summary tables.")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    output_dir = make_output_dir(args.output_dir)
    df = pd.read_csv(args.data, sep=args.sep)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available columns: {list(df.columns)}")

    numeric_cols, categorical_cols = infer_feature_types(df, args.target)
    structure = basic_structure_table(df)
    unknowns = unknown_token_table(df, categorical_cols)
    numeric_summary = numeric_summary_table(df, numeric_cols)
    target_dist = target_summary(df, args.target)
    numeric_rel = numeric_target_relationships(df, numeric_cols, args.target)
    categorical_rel = categorical_target_relationships(df, categorical_cols, args.target)
    leakage_risks = leakage_risk_table(df)
    issues = build_issue_table(structure, unknowns, numeric_rel, categorical_rel)

    screening_summary = pd.concat(
        [
            numeric_rel[
                ["feature", "tool", "task", "std_mean_diff", "pearson_with_target", "mannwhitney_pvalue"]
            ].rename(columns={"std_mean_diff": "association_strength"}),
            categorical_rel[
                ["feature", "tool", "task", "cramers_v", "chi2_pvalue", "top_lift_level_rate"]
            ].rename(columns={"cramers_v": "association_strength", "chi2_pvalue": "mannwhitney_pvalue"}),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values("association_strength", key=lambda s: s.abs(), ascending=False)

    structure.to_csv(output_dir / "structure_table.csv", index=False)
    unknowns.to_csv(output_dir / "unknown_token_table.csv", index=False)
    numeric_summary.to_csv(output_dir / "numeric_summary.csv", index=False)
    target_dist.to_csv(output_dir / "target_distribution.csv", index=False)
    numeric_rel.to_csv(output_dir / "numeric_target_relationships.csv", index=False)
    categorical_rel.to_csv(output_dir / "categorical_target_relationships.csv", index=False)
    screening_summary.to_csv(output_dir / "screening_summary_table.csv", index=False)
    issues.to_csv(output_dir / "data_quality_issues.csv", index=False)
    leakage_risks.to_csv(output_dir / "potential_leakage_risks.csv", index=False)

    plot_target_distribution(df, args.target, output_dir)
    plot_missing_and_unknown(structure, unknowns, output_dir)
    plot_numeric_vs_target(df, numeric_rel, args.target, output_dir)
    plot_target_rate_by_categorical(df, categorical_rel, args.target, output_dir)
    plot_correlation_heatmap(df, numeric_cols, output_dir)

    findings = summarise_findings(target_dist, unknowns, numeric_rel, categorical_rel)

    print_section("EDA plan")
    print("1. Validate dataset structure and infer numeric/categorical feature groups.")
    print("2. Quantify missingness, encoded missing tokens, and target imbalance.")
    print("3. Screen predictor-target relationships using separate numeric and categorical tests.")
    print("4. Save compact plots and summary tables for modelling handoff.")

    print_section("Dataset overview")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target: {args.target}")
    print("\nVariable structure (top 10 by missingness / cardinality):")
    print(structure.head(10).to_string(index=False))

    print_section("Summary statistics")
    print(numeric_summary.round(3).head(10).to_string(index=False))

    print_section("Target distribution")
    print(target_dist.round(3).to_string(index=False))

    print_section("Relationship summary table")
    print(screening_summary.round(4).head(12).fillna("").to_string(index=False))

    print_section("Data quality issues")
    if issues.empty:
        print("No explicit missing-value or dominance issues flagged by the configured thresholds.")
    else:
        print(issues.head(12).to_string(index=False))

    print_section("Potential leakage risks")
    print(leakage_risks.to_string(index=False))

    print_section("Key findings")
    for i, finding in enumerate(findings, start=1):
        print(f"{i}. {finding}")

    print_section("Modelling implications")
    print("1. Use stratified validation and imbalance-aware metrics such as PR-AUC, balanced accuracy, or class-weighted log loss.")
    print("2. Treat 'unknown' as its own category or explicit missing indicator rather than silently dropping rows.")
    print("3. Train one model with all predictors and one without duration/history-at-risk fields to test leakage-sensitive performance.")

    print_section("Artifacts")
    print(f"Saved tables and plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
