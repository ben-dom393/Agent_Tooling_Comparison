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
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
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


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def metric_table(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray, model_name: str) -> dict[str, float | str]:
    return {
        "model": model_name,
        "task": "binary_classification",
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "positive_rate_pred": float(np.mean(y_pred)),
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Baseline Logistic Regression Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticklabels(["no", "yes"])
    ax.set_yticklabels(["no", "yes"], rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple, reproducible baseline model for bank marketing classification.")
    parser.add_argument("--data", default="bank-additional-full.csv", help="Path to the input CSV.")
    parser.add_argument("--target", default="y", help="Target column.")
    parser.add_argument("--sep", default=";", help="CSV delimiter.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test fraction.")
    parser.add_argument("--output-dir", default="outputs/baseline", help="Directory for result artifacts.")
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["duration"],
        help="Columns to drop before modelling to avoid leakage or unavailable features.",
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

    y = working_df[args.target].map({"no": 0, "yes": 1})
    X = working_df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=SEED,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    dummy = DummyClassifier(strategy="prior")
    dummy_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", dummy),
        ]
    )

    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
    )
    logistic_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", logistic),
        ]
    )

    dummy_pipe.fit(X_train, y_train)
    logistic_pipe.fit(X_train, y_train)

    dummy_pred = dummy_pipe.predict(X_test)
    dummy_score = dummy_pipe.predict_proba(X_test)[:, 1]
    log_pred = logistic_pipe.predict(X_test)
    log_score = logistic_pipe.predict_proba(X_test)[:, 1]

    results = pd.DataFrame(
        [
            metric_table(y_test, dummy_pred, dummy_score, "DummyClassifier(prior)"),
            metric_table(y_test, log_pred, log_score, "LogisticRegression(balanced)"),
        ]
    ).sort_values("pr_auc", ascending=False)

    report = classification_report(y_test, log_pred, target_names=["no", "yes"], output_dict=True)
    report_df = pd.DataFrame(report).T.reset_index(names="label")

    results.to_csv(output_dir / "baseline_results.csv", index=False)
    report_df.to_csv(output_dir / "logistic_classification_report.csv", index=False)
    plot_confusion_matrix(y_test, log_pred, output_dir / "logistic_confusion_matrix.png")

    print_section("Modelling plan")
    print("1. Treat the problem as binary classification with target 'y'.")
    print("2. Use a stratified train/test split for a simple, reproducible evaluation harness.")
    print("3. Drop duration by default to reduce obvious leakage risk.")
    print("4. Compare a dummy prior classifier to a regularised logistic regression pipeline.")

    print_section("Prediction type and metrics")
    print("Prediction type: binary classification")
    print("Primary metrics: PR-AUC and ROC-AUC")
    print("Support metric: balanced accuracy")
    print("Reason: the positive class is only about 11.3%, so PR-AUC is more informative than plain accuracy.")

    print_section("Split summary")
    print(f"Random seed: {SEED}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Dropped columns: {args.drop_columns}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    print_section("Reported metric(s)")
    print(results.round(4).to_string(index=False))

    print_section("Logistic classification report")
    print(report_df.round(4).to_string(index=False))

    print_section("Baseline justification")
    print(
        "This is an appropriate baseline because it uses standard preprocessing, a transparent linear model, "
        "and a leakage-aware split without feature engineering or hyperparameter tuning."
    )
    print(
        "It is intentionally simple: the goal is to establish a credible reference point, not to maximise leaderboard performance."
    )

    print_section("Artifacts")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
