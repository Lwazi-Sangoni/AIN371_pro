"""Gemini API client for grounded academic risk explanations."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.assistant.local_explainer import build_local_explanation
from src.assistant.prompts import (
    EXPLANATION_PROMPT,
    GENERAL_CHAT_PROMPT,
    SYSTEM_PERSONA,
)
from src.assistant.response_cache import get_cached, make_cache_key, set_cached
from src.config import GEMINI_MODEL, PROJECT_ROOT
from src.models.predict import RiskPrediction

load_dotenv(PROJECT_ROOT / ".env")

QUOTA_NOTE = (
    "Gemini daily quota reached — showing a built-in explanation instead. "
    "The quota resets daily, or you can wait about a minute between requests."
)


class GeminiAssistant:
    """Wraps the Gemini API with caching and safe fallbacks."""

    def __init__(self, api_key: str | None = None, model_name: str = GEMINI_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._model = None
        self.last_source = "local"

        if self.api_key and self.api_key != "your_gemini_api_key_here":
            self._init_client()

    def _init_client(self) -> None:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(model_name=self.model_name)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def explain_prediction(
        self,
        prediction: RiskPrediction,
        user_question: str = "Please explain this student's academic risk prediction.",
    ) -> str:
        cache_key = make_cache_key(
            prediction.student_id, prediction.features, user_question
        )
        cached = get_cached(cache_key)
        if cached:
            self.last_source = "cache"
            return cached

        if not self.is_available:
            self.last_source = "local"
            return build_local_explanation(prediction)

        prompt = EXPLANATION_PROMPT.format(
            student_id=prediction.student_id,
            module_code=prediction.module_code,
            predicted_risk=prediction.predicted_risk.upper(),
            confidence_pct=round(prediction.confidence * 100, 1),
            probabilities=", ".join(
                f"{k}: {v * 100:.1f}%" for k, v in prediction.probabilities.items()
            ),
            features_text=self._format_features(prediction),
            factors_text=self._format_factors(prediction),
            user_question=user_question,
        )

        result = self._generate(f"{SYSTEM_PERSONA}\n\n{prompt}", prediction)
        if self.last_source == "gemini":
            set_cached(cache_key, result)
        return result

    def answer_followup(
        self,
        prediction: RiskPrediction,
        user_question: str,
    ) -> str:
        cache_key = make_cached_key_followup(prediction, user_question)
        cached = get_cached(cache_key)
        if cached:
            self.last_source = "cache"
            return cached

        if not self.is_available:
            self.last_source = "local"
            return build_local_explanation(prediction)

        context = (
            f"Student {prediction.student_id}, module {prediction.module_code}, "
            f"predicted risk: {prediction.predicted_risk} "
            f"({prediction.confidence * 100:.1f}% confidence)."
        )
        prompt = GENERAL_CHAT_PROMPT.format(
            prediction_context=context,
            user_question=user_question,
        )
        result = self._generate(f"{SYSTEM_PERSONA}\n\n{prompt}", prediction)
        if self.last_source == "gemini":
            set_cached(cache_key, result)
        return result

    def _generate(self, prompt: str, prediction: RiskPrediction) -> str:
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2048,
                },
            )
            text = self._extract_text(response)
            if text and not text.startswith("Gemini returned"):
                self.last_source = "gemini"
                return text
            self.last_source = "local"
            return build_local_explanation(prediction)
        except Exception as exc:
            error_text = str(exc)
            if "429" in error_text or "quota" in error_text.lower():
                self.last_source = "local"
                return build_local_explanation(prediction, quota_note=QUOTA_NOTE)
            self.last_source = "local"
            return build_local_explanation(
                prediction,
                quota_note=f"Gemini unavailable ({error_text[:120]}). Showing built-in explanation.",
            )

    def _extract_text(self, response) -> str:
        if not response.candidates:
            return ""

        parts = []
        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    parts.append(part.text)

        return "\n".join(parts).strip()

    @staticmethod
    def _format_features(prediction: RiskPrediction) -> str:
        labels = {
            "attendance_pct": "Attendance (%)",
            "formative_avg": "Formative assessment average (%)",
            "missed_submissions": "Missed submissions (count)",
            "moodle_logins_per_week": "Moodle logins per week",
            "quiz_avg": "Quiz average (%)",
            "prev_module_avg": "Previous module average (%)",
            "assignment_completion_pct": "Assignment completion (%)",
        }
        return "\n".join(
            f"- {labels.get(k, k)}: {v}" for k, v in prediction.features.items()
        )

    @staticmethod
    def _format_factors(prediction: RiskPrediction) -> str:
        if not prediction.top_factors:
            return "- No major concern thresholds were crossed."
        return "\n".join(
            f"- {f['label']}: {f['value']} (concern: {f['concern']})"
            for f in prediction.top_factors
        )


def make_cached_key_followup(prediction: RiskPrediction, question: str) -> str:
    return make_cache_key(
        f"{prediction.student_id}:followup",
        prediction.features,
        question,
    )
