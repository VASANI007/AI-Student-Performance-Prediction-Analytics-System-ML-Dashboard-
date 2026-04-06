import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def preprocess_train(input_path="data/raw/student_data.csv"):
    df = pd.read_csv(input_path)

    # Missing values
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop unnecessary
    df = df.drop(columns=['student_id', 'student_name'], errors='ignore')

    # Encoding
    df = pd.get_dummies(df, drop_first=True)

    return df


def preprocess_predict(df):
    df = df.copy()
    df = df.drop(columns=['student_id', 'student_name'], errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    return df