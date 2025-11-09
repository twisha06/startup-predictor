
Finalized Startup Success Prediction Project
-------------------------------------------

Target column: 'Status'
Mapping: 'acquired' -> 1 (success), everything else -> 0 (failure)

Folder structure:
- data/                 (put your 'startup data.csv' here)
- src/
  - preprocess.py
  - model_train.py
  - explain.py
  - api.py
- model/                (trained models will be saved here)
- run_training.sh
- requirements.txt
- Dockerfile

How to run (recommended):
1. Move your dataset 'startup data.csv' into the data/ folder.
2. Create a virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
3. Train the model:
   bash run_training.sh
   (This will create model/preprocessor.joblib and model/model.joblib)
4. Run the API:
   uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
   Then open http://127.0.0.1:8000/docs to interact.
