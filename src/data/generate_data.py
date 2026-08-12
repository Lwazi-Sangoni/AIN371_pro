"""
Generate synthetic student academic data for Belgium Campus-style modules.

This simulates realistic patterns: students with low attendance and poor
formative marks tend toward higher risk levels. The dataset is for
demonstration and coursework only — not real student records.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, RAW_DATA_DIR, RAW_DATA_PATH, RISK_LABELS


def _compute_risk_score(row: pd.Series) -> float:
    """Derive a continuous risk score from academic indicators."""
    score = 0.0
    score += (100 - row["attendance_pct"]) * 0.25
    score += (100 - row["formative_avg"]) * 0.30
    score += row["missed_submissions"] * 8
    score += max(0, 5 - row["moodle_logins_per_week"]) * 3
    score += (100 - row["quiz_avg"]) * 0.20
    score += (100 - row["prev_module_avg"]) * 0.15
    score += (100 - row["assignment_completion_pct"]) * 0.10
    return score


def _score_to_risk(score: float) -> str:
    if score < 35:
        return "low"
    if score < 65:
        return "moderate"
    return "high"


def generate_dataset(n_students: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    modules = ["AIN371", "PRG381", "DBS371", "MAT371", "NWD371"]
    programmes = ["BICT", "BICT-Hons", "DipICT"]

    for i in range(n_students):
        latent_ability = rng.normal(65, 15)

        attendance_pct = float(np.clip(rng.normal(latent_ability + 5, 12), 30, 100))
        formative_avg = float(np.clip(rng.normal(latent_ability, 14), 20, 100))
        missed_submissions = int(np.clip(rng.poisson(max(0, (100 - latent_ability) / 25)), 0, 6))
        moodle_logins_per_week = float(np.clip(rng.normal(latent_ability / 12, 2), 0, 15))
        quiz_avg = float(np.clip(rng.normal(latent_ability - 3, 13), 15, 100))
        prev_module_avg = float(np.clip(rng.normal(latent_ability, 12), 25, 100))
        assignment_completion_pct = float(np.clip(rng.normal(latent_ability + 8, 10), 40, 100))

        row = {
            "student_id": f"BC{i + 1:06d}",
            "module_code": rng.choice(modules),
            "programme": rng.choice(programmes),
            "attendance_pct": round(attendance_pct, 1),
            "formative_avg": round(formative_avg, 1),
            "missed_submissions": missed_submissions,
            "moodle_logins_per_week": round(moodle_logins_per_week, 1),
            "quiz_avg": round(quiz_avg, 1),
            "prev_module_avg": round(prev_module_avg, 1),
            "assignment_completion_pct": round(assignment_completion_pct, 1),
        }

        risk_score = _compute_risk_score(pd.Series(row))
        row["risk_level"] = _score_to_risk(risk_score)

        # Add realistic noise so the problem is not perfectly separable
        if rng.random() < 0.08:
            row["risk_level"] = rng.choice(RISK_LABELS)

        records.append(row)

    return pd.DataFrame(records)


def main(output_path: Path | None = None, n_students: int = 500) -> Path:
    output_path = output_path or RAW_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(n_students=n_students)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} student records -> {output_path}")
    print(df["risk_level"].value_counts().to_string())
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic student academic data")
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--n-students", type=int, default=500)
    args = parser.parse_args()
    main(args.output, args.n_students)
