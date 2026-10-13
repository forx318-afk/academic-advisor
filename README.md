# Academic Advisor System

Hybrid AI + software-engineering project: a Random Forest predicts a student's
academic standing (At Risk / Good Standing / Excellent) and a rule-based layer
turns the inputs into concrete advice. Served with Flask.

## Workflow
1. `notebooks/train_model.ipynb` - trained on Google Colab, exports `advisor_model.pkl`
2. `models/advisor_model.pkl` - the deployed model
3. `app.py` - Flask API + web UI
4. `tests/` - pytest suite

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python app.py          # http://127.0.0.1:5000
```

## Deploy (Render / Railway)
Start command: `gunicorn app:app`
