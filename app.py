"""Academic Advisor - Flask app serving the pickled model."""
import pickle
from pathlib import Path
from flask import Flask, jsonify, render_template, request

MODEL_PATH = Path(__file__).parent / "models" / "advisor_model.pkl"
with open(MODEL_PATH, "rb") as f:
    BUNDLE = pickle.load(f)
MODEL, FEATURES, LABELS = BUNDLE["model"], BUNDLE["features"], BUNDLE["labels"]

app = Flask(__name__)


def advise(d: dict) -> list[str]:
    """Rule-based layer: turns raw inputs into concrete advice."""
    tips = []
    if d["attendance"] < 75:
        tips.append("Raise attendance above 75% - many courses bar students below it from exams.")
    if d["study_hours"] < 12:
        tips.append("Aim for at least 12-15 focused study hours per week.")
    if d["assignments"] < 70:
        tips.append("Submit more assignments on time; they are the easiest marks to gain.")
    if d["sleep_hours"] < 6:
        tips.append("Sleep 6-8 hours; memory and concentration drop sharply below that.")
    if d["extra_hours"] > 15:
        tips.append("Extra-curricular/side work is heavy; protect time for coursework.")
    if d["failed_courses"] > 0:
        tips.append("Plan carryover courses early and see your course adviser.")
    return tips or ["Keep up the current habits - you are on a strong track."]


def parse(payload: dict) -> dict:
    missing = [f for f in FEATURES if payload.get(f) in (None, "")]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    return {f: float(payload[f]) for f in FEATURES}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    try:
        data = parse(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    row = [[data[f] for f in FEATURES]]
    idx = int(MODEL.predict(row)[0])
    proba = MODEL.predict_proba(row)[0]
    return jsonify(
        standing=LABELS[idx],
        confidence=round(float(proba.max()), 3),
        advice=advise(data),
    )


if __name__ == "__main__":
    app.run(debug=True)
