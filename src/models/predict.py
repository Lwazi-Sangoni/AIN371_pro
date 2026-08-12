"""Run predictions with the trained academic risk model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH, RISK_LABELS
from src.data.preprocess import student_record_to_features


@dataclass
class RiskPrediction:
    student_id: str
    module_code: str
    predicted_risk: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float]
    top_factors: list[dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)


def load_model(path=None):
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python -m src.models.train"
        )
    return joblib.load(path)


def _describe_top_factors(features: dict[str, float], prediction: str) -> list[dict[str, Any]]:
    """Rule-based factor summary aligned with model features (for grounding)."""
    factors = []

    checks = [
        ("attendance_pct", "Attendance", lambda v: v < 70, "below 70%", "low"),
        ("formative_avg", "Formative assessment average", lambda v: v < 50, "below 50%", "low"),
        ("missed_submissions", "Missed submissions", lambda v: v >= 2, "2 or more", "high"),
        ("moodle_logins_per_week", "Moodle activity", lambda v: v < 3, "below 3 logins/week", "low"),
        ("quiz_avg", "Quiz average", lambda v: v < 50, "below 50%", "low"),
        ("prev_module_avg", "Previous module average", lambda v: v < 50, "below 50%", "low"),
        ("assignment_completion_pct", "Assignment completion", lambda v: v < 75, "below 75%", "low"),
    ]

    for key, label, condition, threshold_text, direction in checks:
        value = features[key]
        if condition(value):
            factors.append(
                {
                    "feature": key,
                    "label": label,
                    "value": value,
                    "concern": threshold_text,
                    "direction": direction,
                }
            )

    return factors[:4]


def predict_student(record: dict, pipeline=None) -> RiskPrediction:
    pipeline = pipeline or load_model()

    features_df = student_record_to_features(record)
    features = features_df.iloc[0].to_dict()

    predicted = pipeline.predict(features_df)[0]
    proba = pipeline.predict_proba(features_df)[0]
    classes = list(pipeline.named_steps["classifier"].classes_)

    prob_dict = {cls: round(float(p), 3) for cls, p in zip(classes, proba)}
    confidence = prob_dict[predicted]

    return RiskPrediction(
        student_id=str(record.get("student_id", "Unknown")),
        module_code=str(record.get("module_code", "Unknown")),
        predicted_risk=predicted,
        confidence=confidence,
        probabilities=prob_dict,
        features={k: round(float(v), 2) for k, v in features.items()},
        top_factors=_describe_top_factors(features, predicted),
    )


def predict_from_dataframe(df: pd.DataFrame, pipeline=None) -> list[RiskPrediction]:
    pipeline = pipeline or load_model()
    return [
        predict_student(row.to_dict(), pipeline=pipeline)
        for _, row in df.iterrows()
    ]
