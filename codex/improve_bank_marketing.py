from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42


def make_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def infer_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]
    return numeric_cols, categorical_cols


def replace_unknown_tokens(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    clean = df.copy()
    for col in categorical_cols:
        clean[col] = clean[col].replace("unknown", np.nan)
    return clean


def build_logistic_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def build_tree_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "task": "binary_classification",
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "positive_rate_pred": float(np.mean(y_pred)),
    }
    report = pd.DataFrame(
        classification_report(y_test, y_pred, target_names=["no", "yes"], output_dict=True)
    ).T.reset_index(names="label")
    report.insert(0, "model", name)
    return metrics, report


def plot_metric_comparison(results: pd.DataFrame, output_path: Path) -> None:
    plot_df = results.melt(
        id_vars=["model"],
        value_vars=["roc_auc", "pr_auc", "balanced_accuracy"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Baseline vs Improved Model Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair improvement over the logistic regression baseline.")
    parser.add_argument("--data", default="bank-additional-full.csv", help="Path to the input CSV.")
    parser.add_argument("--target", default="y", help="Target column.")
    parser.add_argument("--sep", default=";", help="CSV delimiter.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test fraction.")
    parser.add_argument("--output-dir", default="outputs/improved", help="Directory for saved artifacts.")
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["duration"],
        help="Columns dropped before modelling for leakage control.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    output_dir = make_output_dir(args.output_dir)

    df = pd.read_csv(args.data, sep=args.sep)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")

    working_df = df.drop(columns=[c for c in args.drop_columns if c in df.columns]).copy()
    numeric_cols, categorical_cols = infer_feature_types(working_df, args.target)
    working_df = replace_unknown_tokens(working_df, categorical_cols)

    X = working_df.drop(columns=[args.target])
    y = working_df[args.target].map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=SEED,
        stratify=y,
    )

    logistic_pipe = Pipeline(
        steps=[
            ("preprocess", build_logistic_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    random_forest_pipe = Pipeline(
        steps=[
            ("preprocess", build_tree_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )

    baseline_metrics, baseline_report = evaluate_model(
        "LogisticRegression(balanced)", logistic_pipe, X_train, X_test, y_train, y_test
    )
    improved_metrics, improved_report = evaluate_model(
        "RandomForest(balanced_subsample)", random_forest_pipe, X_train, X_test, y_train, y_test
    )

    results = pd.DataFrame([baseline_metrics, improved_metrics]).sort_values("pr_auc", ascending=False)
    report_df = pd.concat([baseline_report, improved_report], ignore_index=True)
    plot_metric_comparison(results, output_dir / "metric_comparison.png")

    results.to_csv(output_dir / "baseline_vs_improved_results.csv", index=False)
    report_df.to_csv(output_dir / "classification_reports.csv", index=False)

    print_section("Diagnosis of baseline weaknesses")
    print("1. Logistic regression is linear in the transformed features, so it can miss nonlinear effects and feature interactions.")
    print("2. The dataset mixes macroeconomic variables and categorical campaign/customer fields, which often interact in ways a linear decision boundary cannot capture.")
    print("3. Class imbalance is material, so probability ranking matters; a model with better nonlinear separation can improve PR-AUC even if the split stays unchanged.")

    print_section("Proposed improvements")
    print("1. Replace the linear model with a tree ensemble that can learn nonlinearities and interactions.")
    print("2. Keep the same leakage-aware feature set and split, but use a model less sensitive to monotonic linear assumptions.")
    print("3. If needed later, tune the decision threshold after model selection, but do not change it during the fair baseline comparison.")

    print_section("Final chosen approach")
    print("Use a RandomForestClassifier with the same train/test split, the same dropped leakage-prone column (`duration`), and the same reported metrics.")
    print("This keeps the comparison fair while changing only the modelling family and the minimal preprocessing it needs.")

    print_section("Baseline vs improved results")
    print(results.round(4).to_string(index=False))

    print_section("Class-wise report for improved model")
    print(
        improved_report.round(4).to_string(index=False)
    )

    print_section("Risk checks")
    print("1. Leakage control is unchanged from the baseline: `duration` is still excluded, and all preprocessing is fit on the training split only.")
    print("2. Overfitting risk is higher for tree ensembles than for logistic regression; `min_samples_leaf=5` is included to reduce variance.")
    print("3. The random forest may improve ranking metrics but produce different class balances at the default 0.5 threshold, so threshold tuning should remain a separate step.")

    print_section("Artifacts")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
