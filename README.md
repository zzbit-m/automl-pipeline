# AutoML Pipeline

Upload a CSV, auto-train 5 models, pick the best one, download predictions. No ML experience needed.

## Quick start

```sh
pip install -e .
streamlit run frontend/app.py
```

Or with Docker:

```sh
docker build -t automl .
docker run -p 8501:8501 automl
```

Open [http://localhost:8501](http://localhost:8501).

## How it works

```
Upload CSV → pick target → auto-detect problem type (classification/regression)
  → preprocess data (fill missing values, scale numbers, encode categories)
  → train 5 models via 5-fold cross-validation
  → evaluate best model on held-out 20%
  → refit best model on all data
  → upload new CSV → download predictions
```

### Models

| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Random Forest | Ridge |
| KNN | Random Forest |
| XGBoost | XGBoost |
| LightGBM | LightGBM |

## Project structure

```
├── frontend/app.py      Streamlit UI (single page)
├── pipeline/            ML pipeline modules
│   ├── ingestion.py     CSV loading & validation
│   ├── detector.py      Auto-detect classification vs regression
│   ├── preprocessing.py ColumnTransformer (impute, scale, encode)
│   ├── trainer.py       Cross-validation model comparison
│   ├── predictor.py     Predict on new data
│   ├── summary.py       Data summary statistics
│   └── importance.py    Feature importance via permutation
├── db/                  SQLite persistence
│   ├── schema.py        Table definitions
│   └── queries.py       Insert & query operations
├── data/                SQLite database file
├── requirements.txt
├── Dockerfile
└── pyproject.toml
```

## Requirements

- Python >= 3.10
- Dependencies: scikit-learn, xgboost, lightgbm, pandas, streamlit, numpy, altair, joblib

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file to `frontend/app.py`
