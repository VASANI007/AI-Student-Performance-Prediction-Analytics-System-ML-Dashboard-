# src/predict.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pickle

from src.data_preprocessing import preprocess_predict
from src.feature_engineering import create_new_features


def predict(input_data):

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/columns.pkl", "rb") as f:
        columns = pickle.load(f)

    df = pd.DataFrame([input_data])
    df = preprocess_predict(df)
    df = create_new_features(df)
    df = df.reindex(columns=columns, fill_value=0)
    prediction = model.predict(df)

    return round(prediction[0], 2)