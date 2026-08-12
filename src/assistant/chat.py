"""High-level assistant interface combining ML predictions and Gemini explanations."""

from __future__ import annotations

from dataclasses import dataclass

from src.assistant.gemini_client import GeminiAssistant
from src.data.preprocess import load_raw_data
from src.models.predict import RiskPrediction, load_model, predict_student
from src.validation import ValidationError, validate_question, validate_student_id, validate_student_record


@dataclass
class AssistantResponse:
    prediction: RiskPrediction
    explanation: str
    gemini_enabled: bool
    explanation_source: str = "local"  # gemini | cache | local


class AcademicRiskAssistant:
    def __init__(self):
        self.model = load_model()
        self.gemini = GeminiAssistant()
        self._students = None

    @property
    def students(self):
        if self._students is None:
            self._students = load_raw_data()
        return self._students

    def get_student_ids(self) -> list[str]:
        return sorted(self.students["student_id"].unique().tolist())

    def get_student_record(self, student_id: str) -> dict:
        student_id = validate_student_id(student_id)
        matches = self.students[self.students["student_id"] == student_id]
        if matches.empty:
            raise ValidationError(
                f"Student '{student_id}' was not found in the dataset. "
                "Use manual entry or choose a valid student ID."
            )
        return matches.iloc[0].to_dict()

    def analyse(
        self,
        record: dict,
        question: str = "Please explain this student's academic risk prediction.",
    ) -> AssistantResponse:
        record = validate_student_record(record)
        question = validate_question(question)

        prediction = predict_student(record, pipeline=self.model)
        explanation = self.gemini.explain_prediction(prediction, question)
        return AssistantResponse(
            prediction=prediction,
            explanation=explanation,
            gemini_enabled=self.gemini.is_available,
            explanation_source=self.gemini.last_source,
        )

    def ask_about_student(
        self,
        student_id: str,
        question: str,
    ) -> AssistantResponse:
        student_id = validate_student_id(student_id)
        question = validate_question(question)
        record = self.get_student_record(student_id)
        prediction = predict_student(record, pipeline=self.model)

        if question.lower().strip() in {
            "explain",
            "explain this",
            "explain the prediction",
            "what does this mean",
        }:
            explanation = self.gemini.explain_prediction(prediction, question)
        else:
            explanation = self.gemini.answer_followup(prediction, question)

        return AssistantResponse(
            prediction=prediction,
            explanation=explanation,
            gemini_enabled=self.gemini.is_available,
            explanation_source=self.gemini.last_source,
        )
