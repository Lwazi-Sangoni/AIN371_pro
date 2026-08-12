"""Command-line interface for the Academic Risk Assistant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.chat import AcademicRiskAssistant
from src.validation import ValidationError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Belgium Campus Academic Risk Assistant (CLI)"
    )
    parser.add_argument("--student-id", required=True, help="Student ID e.g. BC000007")
    parser.add_argument(
        "--question",
        default="Please explain this student's academic risk prediction.",
        help="Plain-English question for the assistant",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    assistant = AcademicRiskAssistant()
    try:
        response = assistant.ask_about_student(args.student_id, args.question)
    except ValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(
            json.dumps(
                {
                    "prediction": response.prediction.to_dict(),
                    "explanation": response.explanation,
                    "gemini_enabled": response.gemini_enabled,
                },
                indent=2,
            )
        )
    else:
        pred = response.prediction
        print("=" * 60)
        print(f"Student     : {pred.student_id}")
        print(f"Module      : {pred.module_code}")
        print(f"Risk level  : {pred.predicted_risk.upper()}")
        print(f"Confidence  : {pred.confidence * 100:.1f}%")
        print(f"Probabilities: {pred.probabilities}")
        print("=" * 60)
        print("\nEXPLANATION\n")
        print(response.explanation)
        if not response.gemini_enabled:
            print("\n[Tip: Set GEMINI_API_KEY in .env for Gemini-powered explanations]")


if __name__ == "__main__":
    main()
