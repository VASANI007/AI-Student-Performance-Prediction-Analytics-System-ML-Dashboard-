# src/predict.py

import sys
import os
import pandas as pd
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_predict
from src.feature_engineering import create_new_features
from src.rule_engine import calculate_rule_score


# ✅ Grade Function
def assign_grade(score):
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"


def predict(input_data):
    try:
        model = pickle.load(open("models/model.pkl", "rb"))
        columns = pickle.load(open("models/columns.pkl", "rb"))

        # safe input
        for key in input_data:
            if input_data[key] is None:
                input_data[key] = 0

        # hard rule
        if input_data["study_hours"] == 0 and input_data["attendance"] == 0:
            return {
                "score": 10.0,
                "grade": "D",
                "status": "Very Poor",
                "reasons": ["No academic activity"]
            }

        rule_score, reasons = calculate_rule_score(input_data)

        df = pd.DataFrame([input_data])
        df = preprocess_predict(df)
        df = create_new_features(df)
        df = df.reindex(columns=columns, fill_value=0)

        ml_score = model.predict(df)[0]

        final_score = (0.6 * rule_score) + (0.4 * ml_score)

# 🔥 BOOST FOR TOP STUDENTS
        if (
            input_data['study_hours'] >= 8 and
            input_data['attendance'] >= 85 and
            input_data['previous_score'] >= 80
        ):
            final_score = max(final_score, 85)
        final_score = float(max(0, min(100, final_score)))

        return {
            "score": round(final_score, 2),
            "grade": assign_grade(final_score),
            "status": "Good" if final_score > 70 else "Needs Improvement",
            "reasons": reasons
        }

    except Exception as e:
        return {
            "score": 0.0,
            "grade": "D",
            "status": "Error",
            "reasons": [str(e)]
        }