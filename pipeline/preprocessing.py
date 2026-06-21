from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
import numpy as np
import numpy.typing as npt


class Preprocessor:
    def __init__(self) -> None:
        self._pipeline: ColumnTransformer | None = None
        self._feature_columns: list[str] = []

    def fit(self, df: DataFrame, target: str) -> None:
        X = df.drop(columns=[target])
        self._feature_columns = X.columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers: list[tuple[str, SkPipeline, list[str]]] = []
        if numeric_cols:
            transformers.append(
                ("num", SkPipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), numeric_cols)
            )
        if categorical_cols:
            transformers.append(
                ("cat", SkPipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
            )

        self._pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        self._pipeline.fit(X)

    def transform(self, df: DataFrame, target: str) -> npt.NDArray:
        if self._pipeline is None:
            raise RuntimeError("call fit() before transform()")
        X = df.drop(columns=[target]) if target in df.columns else df
        return self._pipeline.transform(X)

    def fit_transform(self, df: DataFrame, target: str) -> npt.NDArray:
        self.fit(df, target)
        return self.transform(df, target)

    @property
    def pipeline(self) -> ColumnTransformer:
        if self._pipeline is None:
            raise RuntimeError("call fit() first")
        return self._pipeline

    @property
    def feature_columns(self) -> list[str]:
        if not self._feature_columns:
            raise RuntimeError("call fit() first")
        return self._feature_columns
