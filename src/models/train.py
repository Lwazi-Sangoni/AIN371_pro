"""Train the academic risk classification model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH, OUTPUTS_DIR, RISK_LABELS
from src.data.generate_data import main as generate_data
from src.data.preprocess import (
    build_preprocessor,
    load_raw_data,
    prepare_training_data,
    save_preprocessor,
)


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(save_artifacts: bool = True) -> dict:
    df = load_raw_data()
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=RISK_LABELS).tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "risk_labels": RISK_LABELS,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    if save_artifacts:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, MODEL_PATH)
        save_preprocessor(pipeline.named_steps["preprocessor"])

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        _save_confusion_matrix_plot(y_test, y_pred)
        _save_feature_importance_plot(pipeline)

    return metrics


def _save_confusion_matrix_plot(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=RISK_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=RISK_LABELS,
        yticklabels=RISK_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Academic Risk Classifier — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _save_feature_importance_plot(pipeline: Pipeline) -> None:
    classifier = pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(FEATURE_COLUMNS)[order], importances[order], color="#2E86AB")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train academic risk classifier")
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Regenerate synthetic dataset before training",
    )
    args = parser.parse_args()

    if args.generate_data:
        generate_data()

    metrics = train_model()
    print("Training complete.")
    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.3f}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Metrics  : {METRICS_PATH}")


if __name__ == "__main__":
    main()
