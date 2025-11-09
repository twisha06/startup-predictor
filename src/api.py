from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from src.explain import compute_shap, load_artifacts
from fastapi.responses import HTMLResponse

app = FastAPI(title="Startup Success Predictor (final)")
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Startup Success Predictor 🚀</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    background: linear-gradient(135deg, #4f46e5, #3b82f6);
                    color: white;
                    height: 100vh;
                    margin: 0;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                h1 { font-size: 2.5em; margin-bottom: 0.3em; }
                p { font-size: 1.2em; }
                a {
                    color: #ffe58f;
                    text-decoration: none;
                    font-weight: bold;
                }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>🚀 Startup Success Predictor API</h1>
            <p>Welcome! Use the <a href="/docs">/docs</a> page to test predictions.</p>
            <p>Powered by <strong>FastAPI</strong> & Render Cloud</p>
        </body>
    </html>
    """

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