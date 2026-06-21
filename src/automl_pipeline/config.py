from pathlib import Path


UNIQUE_CLASSIFICATION_CUTOFF = 20
CV_FOLDS = 5
HOLDOUT_SIZE = 0.2
RANDOM_STATE = 42
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "pipeline.db"
