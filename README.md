# Belgium Campus — Student Academic Risk Assistant

An AI-powered assistant for **Belgium Campus** that helps lecturers, programme managers, and student-support staff identify students who may benefit from early academic support.

This is **not a chatbot-only project**. A **Random Forest classification model** generates the risk prediction (low / moderate / high). **Google Gemini** then explains that prediction in plain English, grounded in the model's actual output.

> **Important:** Predictions are early-warning indicators only. They must **not** be used as final academic judgements about students. Human staff remain responsible for all decisions.

---

## Problem

Higher-education staff need timely insight into which students may be struggling before final assessments. This assistant:

1. Takes academic indicators (attendance, formative marks, Moodle activity, etc.)
2. Predicts academic risk level using machine learning
3. Lets staff ask questions in plain English
4. Returns a clear, responsible explanation via the Gemini API

---

## Architecture

```
Student data → Random Forest classifier → Risk prediction + probabilities
                                                    ↓
Staff question (plain English) → Gemini API (grounded prompt) → Explanation
```

| Layer | Technology | Role |
|-------|-----------|------|
| ML model | scikit-learn Random Forest | Predict low / moderate / high risk |
| Conversational AI | Google Gemini API | Explain model output responsibly |
| Interface | Streamlit web app + CLI | Non-technical user access |

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (free tier)

### 2. Install dependencies

```bash
cd AIN371_Project
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure Gemini API

```bash
copy .env.example .env
```

Edit `.env` and set your key:

```
GEMINI_API_KEY=your_actual_key_here
```

The app works without Gemini (using a built-in fallback explanation), but Gemini provides much better conversational responses.

### 4. Generate data and train the model

```bash
python -m src.data.generate_data
python -m src.models.train
```

### 5. Run the assistant

**Web app (recommended for demo):**

```bash
streamlit run app/streamlit_app.py
```

**Command line:**

```bash
python app/cli.py --student-id BC0001
python app/cli.py --student-id BC0042 --question "What support would you suggest?"
```

---

## Input features

| Feature | Description |
|---------|-------------|
| `attendance_pct` | Percentage of classes attended |
| `formative_avg` | Average formative assessment mark (%) |
| `missed_submissions` | Number of missed assignment submissions |
| `moodle_logins_per_week` | Average weekly Moodle logins |
| `quiz_avg` | Average quiz mark (%) |
| `prev_module_avg` | Previous module average (%) |
| `assignment_completion_pct` | Percentage of assignments completed |

---

## Prompt engineering & grounding

The Gemini layer is carefully prompted to:

- **Ground responses** in the model's actual prediction, probabilities, and features
- **Never invent** student data or change prediction values
- **Adopt a professional, supportive persona** suitable for academic staff
- **Refuse final decisions** and emphasise human judgement
- **Suggest constructive interventions** (check-ins, tutoring, academic support)

See `src/assistant/prompts.py` for the full prompt templates.

---

## Project structure

```
AIN371_Project/
├── app/
│   ├── streamlit_app.py      # Web interface
│   └── cli.py                # Command-line interface
├── data/
│   └── raw/                  # Synthetic student dataset
├── models/                   # Trained model artifacts
├── notebooks/
│   └── 01_eda_and_training.ipynb
├── outputs/                  # Evaluation plots and metrics
├── src/
│   ├── assistant/            # Gemini client and prompts
│   ├── data/                 # Data generation and preprocessing
│   └── models/               # Training and prediction
├── .env.example
├── requirements.txt
└── README.md
```

---

## Ethics & responsible use

- Data is **synthetic** — no real student records are used
- The model provides **indicators**, not verdicts
- Staff should combine predictions with professional judgement and direct student contact
- Predictions may reflect bias present in training patterns — always review contextually
- Do not use for punitive action without human review

---

## Module alignment (AIN371)

| Requirement | Implementation |
|-------------|----------------|
| Classification / ML technique | Random Forest multi-class classifier |
| Realistic HE problem | Early academic risk at Belgium Campus |
| Non-technical users | Streamlit UI with plain-English questions |
| Model generates prediction | scikit-learn pipeline produces risk + probabilities |
| AI explains model output | Gemini API with grounded prompts |
| Human decision-making | Ethical guardrails in prompts and UI |

---

## Authors

Belgium Campus — AIN371 Project Group
