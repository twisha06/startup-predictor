import joblib
import shap
import numpy as np
import pandas as pd

def load_artifacts(model_path="model/model.joblib", preprocessor_path="model/preprocessor.joblib"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def compute_shap(model, preprocessor, X_raw):
    # X_raw: pandas DataFrame with raw columns (not target)
    X_proc = preprocessor.transform(X_raw)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)
    return shap_values, explainer