from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from src.explain import compute_shap, load_artifacts

app = FastAPI(title="Startup Success Predictor (final)")

class Item(BaseModel):
    data: dict

# load artifacts lazily
MODEL_PATH = "model/model.joblib"
PREPROCESS_PATH = "model/preprocessor.joblib"
model, preprocessor = None, None

def ensure_loaded():
    global model, preprocessor
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts(MODEL_PATH, PREPROCESS_PATH)

@app.post("/predict")
def predict(item: Item):
    ensure_loaded()
    df = pd.DataFrame([item.data])
    # ensure 'Status' isn't used as input by removing if present
    if "Status" in df.columns:
        df = df.drop(columns=["Status"])
    X_proc = preprocessor.transform(df)
    prob = float(model.predict_proba(X_proc)[:,1][0]) if hasattr(model, "predict_proba") else float(model.predict(X_proc)[0])
    # compute shap for explanation of this single row
    try:
        shap_values, explainer = compute_shap(model, preprocessor, df)
        # return basic shape info; client can request detailed arrays
        return {"success_probability": prob, "shap_values_shape": shap_values.shape}
    except Exception as e:
        return {"success_probability": prob, "explain_error": str(e)}