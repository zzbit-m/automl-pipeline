from pathlib import Path
from typing import IO
import pandas as pd
from pandas import DataFrame


class IngestionError(Exception):
    pass


def load_csv(source: str | Path | IO) -> DataFrame:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        if path.suffix.lower() not in (".csv",):
            raise IngestionError(f"Unsupported file type: {path.suffix}")
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(source)
    if df.empty:
        raise IngestionError("CSV is empty")
    if df.columns.tolist() == [""]:
        raise IngestionError("CSV has no named columns")
    return df


def validate_target(df: DataFrame, target: str) -> None:
    if target not in df.columns:
        raise IngestionError(f"Target column '{target}' not found in CSV")
    if df[target].nunique() < 2:
        raise IngestionError(f"Target column '{target}' must have at least 2 unique values")
    if df[target].isnull().any():
        raise IngestionError(f"Target column '{target}' contains missing values")


def validate_inference_columns(infer_df: DataFrame, required: list[str], target: str) -> None:
    required = [c for c in required if c != target]
    missing = [c for c in required if c not in infer_df.columns]
    if missing:
        raise IngestionError(
            f"Inference CSV is missing {len(missing)} column(s) the model was trained on: "
            f"{', '.join(missing)}"
        )
