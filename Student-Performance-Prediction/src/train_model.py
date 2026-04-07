# src/train_model.py

import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import preprocess_train
from src.feature_engineering import feature_engineering_pipeline


def run_training_pipeline(path="data/raw/student_data.csv"):

    print("📥 Loading data...")
    df = preprocess_train(path)

    print("📊 Dataset Shape:", df.shape)

    # ✅ Ensure required columns exist
    required_cols = [
        'previous_score', 'internal_marks', 'assignments',
        'attendance', 'study_hours', 'internet_usage', 'sleep_hours'
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Missing column: {col} → filling with 0")
            df[col] = 0

    # ✅ Fill missing values
    df = df.fillna(0)

    # 🔥 IMPROVED FINAL SCORE FORMULA
    df['final_score'] = (
        0.4 * df['previous_score'] +
        0.3 * df['internal_marks'] +
        0.2 * df['assignments'] +
        0.2 * df['attendance'] +
        0.15 * (df['study_hours'] * 10) -
        0.1 * df['internet_usage'] +
        0.1 * df['sleep_hours']
    )

    # 🔥 Normalize to 0–100 (IMPORTANT)
    max_score = df['final_score'].max()
    if max_score != 0:
        df['final_score'] = (df['final_score'] / max_score) * 100

    # 🔥 Add noise (real-world behavior)
    df['final_score'] += np.random.normal(0, 3, len(df))

    # 🔥 Clamp values
    df['final_score'] = df['final_score'].clip(0, 100)

    print("📊 Final Score Ready")

    # ✅ Feature Engineering
    X, y, columns = feature_engineering_pipeline(df)

    print("📊 Features Shape:", X.shape)
    print("📊 Target Shape:", y.shape)

    # ❌ Prevent empty dataset error
    if len(X) == 0:
        raise ValueError("❌ Dataset empty after preprocessing!")

    # ✅ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Model
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # ✅ Prediction
    y_pred = model.predict(X_test)

    # ✅ Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics_dict = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    print("\n📊 Model Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.3f}")

    # ✅ Save models
    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/columns.pkl", "wb") as f:
        pickle.dump(columns, f)

    with open("models/metrics.pkl", "wb") as f:
        pickle.dump(metrics_dict, f)

    print("\n✅ Model & Metrics Saved Successfully!")


# 🔹 Run directly
if __name__ == "__main__":
    run_training_pipeline()