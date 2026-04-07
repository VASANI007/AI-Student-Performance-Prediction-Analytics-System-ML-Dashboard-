import pandas as pd
import numpy as np

def preprocess_train(path):
    df = pd.read_csv(path)

    df = df.drop(columns=['student_id', 'student_name'], errors='ignore')

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=['gender','parent_education','family_income'], drop_first=True)

    return df


def preprocess_predict(df):
    df = pd.get_dummies(df, drop_first=True)
    return df