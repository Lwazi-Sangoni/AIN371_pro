"""Input validation for student records and staff questions."""

from __future__ import annotations

import re

STUDENT_ID_PATTERN = re.compile(r"^BC\d{6}$")
MODULE_CODE_PATTERN = re.compile(r"^[A-Za-z]{3}\d{3}$")

ACADEMIC_KEYWORDS = (
    "student",
    "risk",
    "support",
    "help",
    "attendance",
    "mark",
    "marks",
    "grade",
    "module",
    "explain",
    "concern",
    "fail",
    "failing",
    "progress",
    "academic",
    "assessment",
    "submission",
    "moodle",
    "quiz",
    "formative",
    "tutor",
    "intervention",
    "learning",
    "performance",
    "struggling",
    "coursework",
    "assignment",
    "engagement",
    "referral",
    "lecturer",
    "class",
    "behind",
    "catch up",
    "warning",
    "at-risk",
)

OFF_TOPIC_PATTERNS = (
    r"\bhow old am i\b",
    r"\bwhat('s| is) my (name|age)\b",
    r"\bwho are you\b",
    r"\btell me a joke\b",
    r"\bweather\b",
    r"\bwhat time is it\b",
    r"\bcapital of\b",
    r"\bwho won\b",
    r"\brecipe\b",
    r"\bfootball\b",
    r"\bsport\b",
)


class ValidationError(ValueError):
    """Raised when user input fails validation."""


def validate_student_id(student_id: str) -> str:
    value = (student_id or "").strip()
    if not value:
        raise ValidationError("Student ID is required.")
    if " " in value:
        raise ValidationError(
            "Student ID must not contain spaces. Expected format: BC followed by 6 digits (e.g. BC000007)."
        )
    if not STUDENT_ID_PATTERN.match(value):
        raise ValidationError(
            f"Invalid student ID '{student_id}'. "
            "Must start with BC followed by exactly 6 numbers, no spaces or special characters "
            "(e.g. BC000007)."
        )
    return value


def validate_module_code(module_code: str) -> str:
    value = (module_code or "").strip()
    if not value:
        raise ValidationError("Module code is required.")
    if " " in value:
        raise ValidationError(
            "Module code must not contain spaces. Expected format: 3 letters + 3 numbers (e.g. AIN371)."
        )
    if not MODULE_CODE_PATTERN.match(value):
        raise ValidationError(
            f"Invalid module code '{module_code}'. "
            "Must be 3 letters followed by 3 numbers, no spaces or special characters "
            "(e.g. AIN371)."
        )
    return value.upper()


def validate_question(question: str) -> str:
    value = (question or "").strip()
    if not value:
        raise ValidationError("Please enter a question about the student's academic progress.")

    lower = value.lower()
    if any(re.search(pattern, lower) for pattern in OFF_TOPIC_PATTERNS):
        raise ValidationError(
            "This question is not about a student's academic progress. "
            "Please ask about the student's risk level, academic performance, or support options."
        )

    if not any(keyword in lower for keyword in ACADEMIC_KEYWORDS):
        raise ValidationError(
            "Please ask a question related to the student's academic risk, performance, "
            "or support (e.g. 'Explain this student's risk and suggest support')."
        )

    return value


def validate_student_record(record: dict) -> dict:
    """Validate and normalise student ID and module code in a record."""
    validated = dict(record)
    validated["student_id"] = validate_student_id(str(record.get("student_id", "")))
    validated["module_code"] = validate_module_code(str(record.get("module_code", "")))
    return validated
