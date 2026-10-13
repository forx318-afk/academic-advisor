# %% [markdown]
# # Academic Advisor - Model Training (run in Google Colab)
# Predicts a student's academic standing (At Risk / Good Standing / Excellent).

# %% 1. Imports
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# %% 2. Create a dataset (synthetic; swap in real, anonymised data if you have it)
rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "study_hours":   rng.uniform(0, 40, n),      # per week
    "attendance":    rng.uniform(30, 100, n),    # %
    "prev_gpa":      rng.uniform(1.0, 5.0, n),   # 5.0 scale
    "assignments":   rng.uniform(20, 100, n),    # % submitted
    "sleep_hours":   rng.uniform(3, 10, n),      # per night
    "extra_hours":   rng.uniform(0, 20, n),      # clubs / side work per week
    "failed_courses": rng.integers(0, 6, n),
})
score = (12 * df.prev_gpa + 0.25 * df.attendance + 0.8 * df.study_hours
         + 0.15 * df.assignments + 1.0 * df.sleep_hours
         - 2.5 * df.failed_courses - 0.3 * df.extra_hours
         + rng.normal(0, 4, n))
lo, hi = score.quantile(0.25), score.quantile(0.80)
df["standing"] = np.select([score < lo, score >= hi], [0, 2], default=1)
LABELS = {0: "At Risk", 1: "Good Standing", 2: "Excellent"}
print(df.standing.map(LABELS).value_counts())

# %% 3. Train / evaluate
FEATURES = list(df.columns[:-1])
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df.standing, test_size=0.2, random_state=42, stratify=df.standing)

model = Pipeline([
    ("scale", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=list(LABELS.values())))

# %% 4. Export the .pkl
with open("advisor_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": FEATURES, "labels": LABELS}, f)

# In Colab, download it:
# from google.colab import files; files.download("advisor_model.pkl")
