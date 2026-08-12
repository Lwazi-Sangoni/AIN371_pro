"""Streamlit web app for the Belgium Campus Academic Risk Assistant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.assistant.chat import AcademicRiskAssistant
from src.config import METRICS_PATH, OUTPUTS_DIR

RISK_COLOURS = {
    "low": "#2ECC71",
    "moderate": "#F39C12",
    "high": "#E74C3C",
}


@st.cache_resource
def get_assistant() -> AcademicRiskAssistant:
    return AcademicRiskAssistant()


def _run_analysis(assistant: AcademicRiskAssistant, record: dict, question: str) -> None:
    from src.validation import ValidationError

    try:
        with st.spinner("Running ML model and generating explanation..."):
            response = assistant.analyse(record, question)
        st.session_state["last_response"] = response
        st.session_state["last_explanation"] = (response.explanation or "").strip()
        st.session_state["last_gemini_enabled"] = response.gemini_enabled
        st.session_state["last_explanation_source"] = response.explanation_source
        st.session_state.pop("last_error", None)
    except ValidationError as exc:
        st.session_state["last_error"] = str(exc)
        st.session_state.pop("last_response", None)
        st.session_state.pop("last_explanation", None)


def main() -> None:
    st.set_page_config(
        page_title="Belgium Campus — Academic Risk Assistant",
        page_icon="🎓",
        layout="wide",
    )

    st.title("Belgium Campus Academic Risk Assistant")
    st.caption(
        "An AI-supported early-warning tool for lecturers and student-support staff. "
        "Predictions are indicators only — not final academic decisions."
    )

    if st.sidebar.button("Clear cache & reload"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

    try:
        assistant = get_assistant()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run these commands first:\n```\npython -m src.data.generate_data\npython -m src.models.train\n```")
        return

    tab_predict, tab_browse, tab_about = st.tabs(
        ["Ask about a student", "Browse dataset", "About the model"]
    )

    with tab_predict:
        _render_prediction_tab(assistant)

    with tab_browse:
        _render_browse_tab(assistant)

    with tab_about:
        _render_about_tab()


def _render_prediction_tab(assistant: AcademicRiskAssistant) -> None:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Student selection")
        mode = st.radio("Input mode", ["Select from dataset", "Enter details manually"])

        if mode == "Select from dataset":
            student_id = st.selectbox("Student ID", assistant.get_student_ids())
            record = assistant.get_student_record(student_id)
            st.json({k: record[k] for k in record if k != "risk_level"})
            actual_risk = record.get("risk_level")
            if actual_risk:
                st.caption(f"Dataset label (for demo only): **{actual_risk}**")
        else:
            st.caption("Format: **BC** + 6 digits (e.g. BC000007). Module: 3 letters + 3 digits (e.g. AIN371).")
            record = {
                "student_id": st.text_input("Student ID", value="BC000007", help="BC followed by 6 numbers, no spaces"),
                "module_code": st.text_input("Module code", value="AIN371", help="3 letters + 3 numbers, e.g. AIN371"),
                "attendance_pct": st.slider("Attendance (%)", 0.0, 100.0, 75.0),
                "formative_avg": st.slider("Formative average (%)", 0.0, 100.0, 55.0),
                "missed_submissions": st.number_input("Missed submissions", 0, 10, 1),
                "moodle_logins_per_week": st.slider("Moodle logins/week", 0.0, 15.0, 4.0),
                "quiz_avg": st.slider("Quiz average (%)", 0.0, 100.0, 60.0),
                "prev_module_avg": st.slider("Previous module average (%)", 0.0, 100.0, 58.0),
                "assignment_completion_pct": st.slider("Assignment completion (%)", 0.0, 100.0, 80.0),
            }

        question = st.text_area(
            "Your question (plain English)",
            value="Please explain this student's academic risk and what support might help.",
            height=100,
        )

        if st.button("Analyse & Explain", type="primary"):
            _run_analysis(assistant, record, question)

    with col_right:
        if st.session_state.get("last_error"):
            st.error(st.session_state["last_error"])
        elif "last_response" in st.session_state:
            _display_response(st.session_state["last_response"])
        else:
            st.info("Select a student and click **Analyse & Explain** to see results here.")


def _display_response(response) -> None:
    pred = response.prediction
    colour = RISK_COLOURS.get(pred.predicted_risk, "#95A5A6")

    st.subheader("Model prediction")
    st.markdown(
        f"<div style='padding:1rem;border-radius:8px;background:{colour}22;"
        f"border-left:4px solid {colour}'>"
        f"<strong>Risk level:</strong> {pred.predicted_risk.upper()}<br>"
        f"<strong>Confidence:</strong> {pred.confidence * 100:.1f}%</div>",
        unsafe_allow_html=True,
    )

    st.write("**Probability breakdown**")
    st.bar_chart(pred.probabilities)

    if pred.top_factors:
        st.write("**Key concern areas**")
        for factor in pred.top_factors:
            st.warning(f"{factor['label']}: {factor['value']} ({factor['concern']})")

    st.subheader("AI explanation")
    source = getattr(response, "explanation_source", "local")
    if source == "gemini":
        st.success("Powered by Gemini (grounded in model output)")
    elif source == "cache":
        st.success("Explanation loaded from cache (saves API quota)")
    else:
        st.info("Built-in explanation (Gemini quota unavailable — still grounded in model output)")

    explanation = st.session_state.get(
        "last_explanation",
        (response.explanation or "").strip(),
    )
    # Horizontal rules can break Streamlit markdown rendering
    explanation = explanation.replace("\n---\n", "\n\n").replace("---", "")

    if explanation:
        with st.container(border=True):
            st.markdown(explanation, unsafe_allow_html=False)
    else:
        st.error(
            "No explanation text was returned. Please click **Analyse & Explain** again."
        )


def _render_browse_tab(assistant: AcademicRiskAssistant) -> None:
    st.subheader("Student dataset preview")
    df = assistant.students
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="student_academic_data.csv",
        mime="text/csv",
    )


def _render_about_tab() -> None:
    st.subheader("How it works")
    st.markdown(
        """
1. **Machine learning model** — A Random Forest classifier predicts whether a student
   may be at **low**, **moderate**, or **high** risk of needing additional academic support.
2. **Conversational AI** — Google Gemini explains the model's output in plain English,
   grounded in the actual prediction and input features.
3. **Human decision-making** — Staff remain responsible for all academic decisions.
   This tool provides early-warning indicators only.

**Input features:** attendance, formative marks, missed submissions, Moodle activity,
quiz results, previous module performance, assignment completion.

**Technique:** Multi-class classification (Random Forest)
        """
    )

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
        st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        st.metric("Macro F1 score", f"{metrics['macro_f1']:.3f}")

    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    fi_path = OUTPUTS_DIR / "feature_importance.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion matrix")
    if fi_path.exists():
        st.image(str(fi_path), caption="Feature importance")


if __name__ == "__main__":
    main()
