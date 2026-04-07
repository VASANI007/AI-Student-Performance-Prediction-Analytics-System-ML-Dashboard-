from src.train_model import run_training_pipeline
from src.predict import predict
import argparse

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
        run_training_pipeline("data/raw/student_data.csv")

    elif args.mode == "predict":

        input_data = {
            "study_hours": 5,
            "attendance": 70,
            "previous_score": 60,
            "assignments": 65,
            "internal_marks": 70,
            "sleep_hours": 6,
            "internet_usage": 4,
            "extra_activities": "Yes",
            "parent_education": "Graduate",
            "family_income": "Medium"
        }

        result = predict(input_data)

        print("\n===== RESULT =====")
        print("Score :", result["score"])
        print("Grade :", result["grade"])
        print("Reasons:", result["reasons"])

if __name__ == "__main__":
    main()