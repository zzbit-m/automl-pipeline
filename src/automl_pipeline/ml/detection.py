from pandas import DataFrame

from automl_pipeline.config import UNIQUE_CLASSIFICATION_CUTOFF


def detect_problem(df: DataFrame, target: str) -> str:
    unique_count = df[target].nunique()
    dtype = df[target].dtype

    if _is_float_like(dtype) and unique_count > UNIQUE_CLASSIFICATION_CUTOFF:
        return "regression"
    if _is_int_like(dtype) and unique_count > UNIQUE_CLASSIFICATION_CUTOFF:
        return "regression"
    if dtype == "object" or dtype.name == "category":
        return "classification"

    if unique_count <= UNIQUE_CLASSIFICATION_CUTOFF:
        return "classification"
    return "regression"


def _is_float_like(dtype) -> bool:
    name = dtype.name
    return "float" in name or name in ("double",)


def _is_int_like(dtype) -> bool:
    name = dtype.name
    return "int" in name or name in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64")
