import pandas as pd
from sklearn.model_selection import train_test_split

def map_status(df, status_col="Status"):
    # Map 'acquired' -> 1, everything else -> 0
    df = df.copy()
    df[status_col] = df[status_col].astype(str).str.strip().str.lower()
    df["target"] = (df[status_col] == "acquired").astype(int)
    return df

def create_features(df):
    df = df.copy()
    # Example derived features
    if "founding_year" in df.columns:
        df["startup_age"] = pd.Timestamp.now().year - pd.to_numeric(df["founding_year"], errors="coerce")
    if "total_funding" in df.columns and "startup_age" in df.columns:
        df["funding_efficiency"] = pd.to_numeric(df["total_funding"], errors="coerce") / df["startup_age"].replace(0,1)
    return df

def get_feature_lists(df, target_col="target"):
    # return numeric and categorical column lists excluding target
    df = df.copy()
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    # remove status column if present
    if "Status" in categorical_cols:
        categorical_cols.remove("Status")
    return numeric_cols, categorical_cols

def load_and_prepare(path_or_df):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()
    df = map_status(df, status_col="Status")
    df = create_features(df)
    return df

def split_df(df, target_col="target", test_size=0.30, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)