# src/train_model.py
import sys
import os
from xml.parsers.expat import model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.evaluate_model import evaluate
from src.data_preprocessing import preprocess_train
from src.feature_engineering import feature_engineering_pipeline


def run_training_pipeline(data_path="data/raw/student_data.csv"):
    df = preprocess_train(data_path)
    df['final_score'] = (
        df['previous_score'] * 0.5 +
        df['internal_marks'] * 0.2 +
        df['assignments'] * 0.2 +
        df['attendance'] * 0.1
    )

    X, y, columns = feature_engineering_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    
    # Prediction on test data
    y_pred = model.predict(X_test)

    # CALL EVALUATION
    evaluate(y_test, y_pred)

    metrics = {
    "MAE": float(mean_absolute_error(y_test, y_pred)),
    "RMSE": float(mean_squared_error(y_test, y_pred) ** 0.5),
    "R2": float(r2_score(y_test, y_pred))
    }

    with open("models/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
        os.makedirs("models", exist_ok=True)
    
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/columns.pkl", "wb") as f:
        pickle.dump(columns, f)

    print("Training Done")