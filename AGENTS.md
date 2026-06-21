# automl-pipeline

Streamlit + scikit-learn + SQLite. Upload CSV → auto-detect problem → train 5 models → pick best → download predictions.

## Project structure

```
automl-pipeline/
├── frontend/      Streamlit pages & widgets
│   └── app.py     Main app: upload, target select, train, leaderboard, download
├── pipeline/      Model training, preprocessing, best-model selection
│   ├── ingestion.py    CSV loading & validation
│   ├── detector.py     Auto-detect classification vs regression
│   ├── preprocessing.py Sklearn ColumnTransformer (impute, encode, scale)
│   ├── trainer.py      Compare models via cross_val_score
│   └── predictor.py    Predict on new CSV using best model
├── db/            SQLite schema & queries
│   ├── schema.py       CREATE TABLE + init_db()
│   └── queries.py      Insert/query jobs, model_results, predictions
├── data/          uploaded CSVs, cached artifacts, pipeline.db
├── doc/           analysis.md (decisions log, status checklist)
├── pyproject.toml
└── AGENTS.md
```

## Key constraints

- **Users are non-technical analysts.** One-click UI, no ML config. No CLI mode.
- **Output is predictions on new data.** Not model artifacts.
- **Problem type is auto-detected** from the CSV target column (classification vs regression).
- **Python >= 3.10** required.

## Stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Single-file app, widgets built-in, analyst-friendly |
| AutoML | scikit-learn + xgboost + lightgbm | PyCaret incompatible with Python 3.13; manual pipeline gives full control |
| DB | SQLite | Zero config, stdlib, enough for single-user |

## Models

**Classification:** Logistic Regression, Random Forest, KNN, XGBoost, LightGBM.  
**Regression:** Linear Regression, Ridge, Random Forest, XGBoost, LightGBM.

Evaluated via 5-fold CV. Metric: accuracy (classification) / RMSE (regression).

## Running

```sh
streamlit run frontend/app.py
```

## Dependencies

```
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
pandas>=2.0
streamlit>=1.28
numpy>=1.24
scipy>=1.11
joblib>=1.3
```
