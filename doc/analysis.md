# Automl-pipeline — Analysis

## Architecture

```
          Upload CSV
              |
              v
      +----------------+
      |  Streamlit UI  |
      | (frontend/)    |
      +----------------+
              |
      User picks target
              |
              v
      +------------------------------------+
      |  Manual sklearn Pipeline           |
      |  (pipeline/)                       |
      |  - auto-detect problem             |
      |  - preprocess (ColumnTransformer)  |
       |  - train 5 models via cross_val    |
      |  - pick best (CV score)            |
      +------------------------------------+
              |
          Predictions
              |
              v
      +----------------+
      |  SQLite DB      |    job history, best model ref
      |  (db/)          |
      +----------------+
```

## File Structure

```
automl-pipeline/
├── data/              uploaded CSVs, cached results, pipeline.db
├── db/                SQLite schema and queries
├── pipeline/          sklearn-based training, preprocessing, prediction
├── frontend/          Streamlit app (single-page)
├── doc/
│   └── analysis.md    this file
├── pyproject.toml     project manifest
```

## Decisions Log

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | ML ecosystem standard |
| UI framework | Streamlit | Lowest overhead for non-technical users; one-file apps, built-in widgets |
| AutoML engine | sklearn (manual pipeline) | PyCaret incompatible with Python 3.13; switched to sklearn + ColumnTransformer + cross_val_score |
| Database | SQLite | Zero-config, ships with Python, enough for single-user / small-team use |
| Problem detection | Auto (heuristic) | User said non-technical analysts — no manual config; dtype + unique-count logic in `detector.py` |
| Output | Predictions on new data | Users want results, not model files |

## Status Checklist

- [x] **Frontend** — Streamlit app with file upload, target selector, train button (`frontend/app.py:20-46`: `st.file_uploader`, `st.selectbox`, `st.button`)
- [x] **Pipeline** — CSV ingestion & validation (`pipeline/ingestion.py`: `load_csv` checks existence/extension/empty; `validate_target` checks presence, >1 unique, no nulls)
- [x] **Pipeline** — Problem auto-detection (`pipeline/detector.py`: `detect_problem` uses dtype + unique-count heuristic)
- [x] **Pipeline** — Auto preprocessing (`pipeline/preprocessing.py`: `Preprocessor` class — `ColumnTransformer` with impute/scale/OHE)
- [x] **Pipeline** — Multi-model training & comparison (`pipeline/trainer.py`: `compare_models` via sklearn `cross_val_score` over 5 class + 5 reg models, incl. XGBoost & LightGBM)
- [x] **Pipeline** — Best model selection & holdout evaluation (`pipeline/trainer.py`: 80/20 split → CV on training → blind evaluation on held-out 20% → refit on all data)
- [x] **Pipeline** — Prediction on user-provided inference CSV (`pipeline/predictor.py`: `predict()` transforms + calls `model.predict()`; wired in `app.py:91-96`)
- [x] **Database** — Job tracking schema (`db/schema.py`: tables `jobs`, `model_results`, `predictions`)
- [x] **Database** — Persist run history & best model metadata (`db/queries.py`: `insert_job`, `insert_model_result`, `insert_predictions`, `get_best_model`)
- [x] **Frontend** — Leaderboard display (`app.py`: shows model name, CV score, std, holdout score)
- [x] **Frontend** — Prediction download (`app.py`: `st.download_button("Download predictions (CSV)")`)
- [x] **Frontend** — Past predictions browse & download from sidebar history
- [x] **Frontend** — Inference column validation before predict (`ingestion.py:validate_inference_columns`)
- [x] **Frontend** — Feature importance bar chart via permutation importance (`pipeline/importance.py`)
- [x] **Frontend** — Download best model as .pkl (`app.py`: `joblib.dumps` + `st.download_button`)
- [x] **Polish** — Smooth progress bar with intermediate steps; `IngestionError` caught separately from generic exceptions
- [x] **Devops** — `requirements.txt` and `Dockerfile` for easy deployment
