# Analysis & Decisions

## Architecture

src-layout (`src/automl_pipeline/`) for modern Python packaging (PEP 517/518).

## Module naming

| Old | New | Reason |
|---|---|---|
| `pipeline/` | `ml/` | Avoids confusion with sklearn's `Pipeline` class |
| `frontend/` | `web/` | Streamlit page is backend-rendered, not client-side |
| `db/schema.py` + `db/queries.py` | `storage/database.py` | Merged — tightly coupled (every query uses schema's connection logic) |
| `detector.py` | `detection.py` | Modules named as nouns, not "-er" agent nouns |
| `trainer.py` | `training.py` | Same reason |
| `predictor.py` | `prediction.py` | Same reason |

## Constants

All centralized in `config.py`:
- `UNIQUE_CLASSIFICATION_CUTOFF` — max unique values to classify (default 20)
- `CV_FOLDS` — cross-validation folds (default 5)
- `HOLDOUT_SIZE` — holdout fraction (default 0.2)
- `RANDOM_STATE` — for reproducibility (default 42)
- `DB_PATH` — SQLite database location

## Key fixes

- **Inference target-column crash** (`preprocessing.py:35`): only drop target column if present in inference CSV.
- **XGBoost/LightGBM CV crash** (`training.py:43-49`): wrap native Booster output in pandas DataFrame when sklearn wrapper returns raw arrays.
- **sys.path fallback** (`web/app.py:3-7`): streamlit run doesn't always honor editable-install `.pth` files, so app.py adds `src/` to `sys.path` at startup.

## Status checklist

- [x] CSV upload and validation
- [x] Problem type auto-detection (classification / regression)
- [x] Preprocessing: impute missing, scale numeric, encode categorical
- [x] 5 models per problem type, trained via 5-fold CV
- [x] Holdout evaluation (80/20 split) before refit on all data
- [x] Leaderboard display with best model selection
- [x] Feature importance (permutation importance)
- [x] Model download as .pkl
- [x] Predict on new CSV with column validation
- [x] Download predictions as CSV
- [x] Data summary stats (overview + per-column + target distribution)
- [x] Past run history in sidebar with browse/download
- [x] Dockerfile
- [x] src-layout restructure
