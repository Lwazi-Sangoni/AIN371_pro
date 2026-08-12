"""Load and preprocess student academic data."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, PREPROCESSOR_PATH, RAW_DATA_PATH, RISK_LABELS


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m src.data.generate_data"
        )
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df[FEATURE_COLUMNS]
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def save_preprocessor(preprocessor: Pipeline, path: Path | None = None) -> None:
    path = path or PREPROCESSOR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path | None = None) -> Pipeline:
    path = path or PREPROCESSOR_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {path}. Run: python -m src.models.train"
        )
    return joblib.load(path)


def student_record_to_features(record: dict) -> pd.DataFrame:
    """Convert a single student dict into a one-row feature DataFrame."""
    missing = [col for col in FEATURE_COLUMNS if col not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return pd.DataFrame([{col: record[col] for col in FEATURE_COLUMNS}])


def validate_risk_label(label: str) -> str:
    label = label.lower().strip()
    if label not in RISK_LABELS:
        raise ValueError(f"Invalid risk label '{label}'. Expected one of {RISK_LABELS}")
    return label
