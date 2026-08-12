"""Generate conversational explanations locally when Gemini is unavailable."""

from __future__ import annotations

from src.models.predict import RiskPrediction

FEATURE_LABELS = {
    "attendance_pct": "attendance",
    "formative_avg": "formative assessment average",
    "missed_submissions": "missed submissions",
    "moodle_logins_per_week": "Moodle activity",
    "quiz_avg": "quiz average",
    "prev_module_avg": "previous module average",
    "assignment_completion_pct": "assignment completion",
}

RISK_OVERVIEW = {
    "low": (
        "{student_id} is progressing well in {module_code} and is currently at **low risk** "
        "of needing urgent academic support ({confidence_pct}% confidence). "
        "This is an early mid-semester check — not a final grade or outcome."
    ),
    "moderate": (
        "{student_id} is currently at **moderate risk** of falling behind in {module_code} "
        "({confidence_pct}% confidence). Some academic indicators suggest the student may "
        "need additional support soon. This is an early warning, not a final result."
    ),
    "high": (
        "{student_id} is currently at **high risk** of struggling in {module_code} "
        "({confidence_pct}% confidence). Several academic indicators are below expected "
        "levels and the student would likely benefit from prompt, supportive intervention. "
        "This is an early warning, not a final academic judgement."
    ),
}

CONCERN_EXPLANATIONS = {
    "attendance_pct": (
        "Attendance is {value}% (below 70%). Missing classes makes it harder to keep up "
        "with new content and reduces opportunities to ask questions in person."
    ),
    "formative_avg": (
        "The formative assessment average is {value}% (below 50%). This suggests gaps in "
        "understanding that may grow if not addressed before summative assessments."
    ),
    "missed_submissions": (
        "The student has {value} missed submission(s). Falling behind on coursework often "
        "leads to further disengagement and lower final marks."
    ),
    "moodle_logins_per_week": (
        "Moodle activity is {value} logins per week (below 3). Low online engagement may "
        "mean the student is missing resources, announcements, or revision material."
    ),
    "quiz_avg": (
        "The quiz average is {value}% (below 50%). Quizzes reflect week-to-week understanding; "
        "a low average suggests core concepts may not yet be secure."
    ),
    "prev_module_avg": (
        "The previous module average was {value}% (below 50%). Past performance can indicate "
        "ongoing study-skill or subject-knowledge challenges."
    ),
    "assignment_completion_pct": (
        "Assignment completion is {value}% (below 75%). Incomplete work limits feedback "
        "opportunities and makes it harder to judge true understanding."
    ),
}

ACTIONS_BY_RISK = {
    "low": [
        "Acknowledge the student's positive progress — brief encouragement can help maintain momentum.",
        "Monitor assignment completion and attendance over the next few weeks in case patterns change.",
    ],
    "moderate": [
        "Arrange a supportive one-on-one check-in to discuss whether time, understanding, or personal factors are affecting progress.",
        "Review recent formative work together and identify specific topics where extra help would make the biggest difference.",
        "Refer the student to academic support or tutoring if gaps in understanding are confirmed.",
    ],
    "high": [
        "Reach out promptly for a compassionate, private meeting - ask what barriers the student is facing (time, health, understanding, access).",
        "Work with the student on a realistic catch-up plan for missed submissions and upcoming assessments.",
        "Refer to student support services and module tutoring; follow up within one week to check progress.",
    ],
}


def build_local_explanation(prediction: RiskPrediction, quota_note: str = "") -> str:
    risk = prediction.predicted_risk.lower()
    confidence_pct = round(prediction.confidence * 100, 1)

    overview = RISK_OVERVIEW.get(risk, RISK_OVERVIEW["moderate"]).format(
        student_id=prediction.student_id,
        module_code=prediction.module_code,
        confidence_pct=confidence_pct,
    )

    if prediction.top_factors:
        concern_lines = []
        for factor in prediction.top_factors:
            key = factor["feature"]
            template = CONCERN_EXPLANATIONS.get(key)
            if template:
                concern_lines.append(f"- {template.format(value=factor['value'])}")
            else:
                concern_lines.append(
                    f"- {factor['label']} is {factor['value']} ({factor['concern']})."
                )
        concerns = "\n".join(concern_lines)
    else:
        concerns = (
            "- No single indicator is critically below threshold, but you should still "
            "review the student's recent work and engagement as the semester continues."
        )

    actions = ACTIONS_BY_RISK.get(risk, ACTIONS_BY_RISK["moderate"])
    action_text = "\n".join(f"- {action}" for action in actions)

    note = ""
    if quota_note:
        note = f"\n\n*{quota_note}*"

    return (
        f"**Overview**\n{overview}\n\n"
        f"**Key concerns**\n{concerns}\n\n"
        f"**What you can do**\n{action_text}"
        f"{note}"
    )
