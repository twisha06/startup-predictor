import os
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

from src.preprocess import get_feature_lists

def build_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")
    return preprocessor

def train_and_save(df, save_dir="model"):
    os.makedirs(save_dir, exist_ok=True)
    # split
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    numeric_cols, categorical_cols = get_feature_lists(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    # Fit/transform training data
    X_train_proc = preprocessor.fit_transform(X_train)
    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_proc, y_train)
    # Train model
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_res, y_res)
    # Save artifacts
    joblib.dump(preprocessor, f"{save_dir}/preprocessor.joblib")
    joblib.dump(model, f"{save_dir}/model.joblib")
    # Evaluate on test set
    X_test_proc = preprocessor.transform(X_test)
    preds = model.predict(X_test_proc)
    proba = model.predict_proba(X_test_proc)[:,1] if hasattr(model, "predict_proba") else None
    print(classification_report(y_test, preds))
    if proba is not None:
        print("ROC AUC:", roc_auc_score(y_test, proba))
    return model, preprocessor