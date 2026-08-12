# Data directory

## Source

The dataset in `raw/student_academic_data.csv` is **synthetically generated** for coursework demonstration. It simulates Belgium Campus-style academic indicators but does **not** contain real student records.

## Regenerate

```bash
python -m src.data.generate_data
```

## Columns

| Column | Type | Description |
|--------|------|-------------|
| student_id | string | Anonymous ID (BC000001, BC000002, …) — BC + 6 digits |
| module_code | string | Module code |
| programme | string | Study programme |
| attendance_pct | float | Attendance percentage |
| formative_avg | float | Formative assessment average |
| missed_submissions | int | Count of missed submissions |
| moodle_logins_per_week | float | Weekly Moodle login average |
| quiz_avg | float | Quiz average mark |
| prev_module_avg | float | Previous module average |
| assignment_completion_pct | float | Assignment completion rate |
| risk_level | string | Target label: low / moderate / high |
