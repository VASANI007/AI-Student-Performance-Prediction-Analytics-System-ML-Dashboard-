# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

from src.train_model import run_training_pipeline
from src.predict import predict


#  Training function
def train():
    print("Running Training Pipeline...\n")
    run_training_pipeline("data/raw/student_data.csv")


#  Prediction function
def make_prediction():
    print("Running Prediction...\n")

    # Example input (you can change values)
    input_data = {
    "study_hours": 9,
    "attendance": 95,
    "previous_score": 90,
    "assignments": 92,
    "internal_marks": 88,
    "sleep_hours": 7,
    "internet_usage": 1,
    "extra_activities": "Yes",
    "gender": "Male",
    "parent_education": "Graduate",
    "family_income": "High"
}

    result = predict(input_data)
    print(f"Predicted Student Score: {result}")


#  Main CLI
def main():
    parser = argparse.ArgumentParser(description="Student Performance Prediction System")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Choose mode: train or predict"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "predict":
        make_prediction()


#  Run
if __name__ == "__main__":
    main()