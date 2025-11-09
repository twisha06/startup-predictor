#!/usr/bin/env bash
set -e
mkdir -p model
python - <<'PY'
from src.preprocess import load_and_prepare, get_feature_lists
from src.model_train import train_and_save
import pandas as pd, os
data_path = "data/startup data.csv"
df = pd.read_csv(data_path)
df = load_and_prepare(df)
numeric_cols, categorical_cols = get_feature_lists(df)
X_train, X_test, y_train, y_test = df['X_train'], df['X_test'], df['y_train'], df['y_test'] if False else (None,None,None,None)
# The train_and_save function handles splitting internally; call it:
train_and_save(df, save_dir="model")
print("Training script finished. Models saved in model/")
PY
