"""Project configuration and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_PATH = RAW_DATA_DIR / "student_academic_data.csv"
MODEL_PATH = MODELS_DIR / "risk_classifier.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"

RISK_LABELS = ["low", "moderate", "high"]

FEATURE_COLUMNS = [
    "attendance_pct",
    "formative_avg",
    "missed_submissions",
    "moodle_logins_per_week",
    "quiz_avg",
    "prev_module_avg",
    "assignment_completion_pct",
]

GEMINI_MODEL = "gemini-flash-latest"
