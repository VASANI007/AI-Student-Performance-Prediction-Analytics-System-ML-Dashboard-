import pandas as pd

def create_new_features(df):
    df = df.copy()
    # Strong derived features
    df['total_academic'] = (
        df['previous_score'] +
        df['assignments'] +
        df['internal_marks']
    ) / 3

    df['performance_index'] = (
        df['study_hours'] * 5 +
        df['attendance'] * 0.5 +
        df['previous_score'] * 0.3
    )

    return df


def feature_engineering_pipeline(df):
    df = create_new_features(df)

    X = df.drop(columns=['final_score'])
    y = df['final_score']

    return X, y, X.columns