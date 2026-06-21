from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import numpy.typing as npt
from pandas import DataFrame

from automl_pipeline.ml.preprocessing import Preprocessor


def _build_feature_map(preprocessor: Preprocessor) -> dict[str, str]:
    pipeline = preprocessor.pipeline
    feature_map: dict[str, str] = {}
    for name, transformer, columns in pipeline.transformers_:
        trans_features = transformer.get_feature_names_out(columns)
        for tf in trans_features:
            matched = [c for c in columns if tf == c or tf.startswith(c + "_")]
            orig = matched[0] if matched else tf
            feature_map[f"{name}__{tf}"] = orig
    return feature_map


def compute_importance(
    model: BaseEstimator,
    X: npt.NDArray,
    y: npt.NDArray,
    preprocessor: Preprocessor,
    metric: str,
) -> DataFrame:
    scoring = "accuracy" if metric == "classification" else "neg_root_mean_squared_error"
    result = permutation_importance(
        model, X, y, n_repeats=10, scoring=scoring, n_jobs=-1, random_state=42
    )

    transformed_names = preprocessor.pipeline.get_feature_names_out()
    feature_map = _build_feature_map(preprocessor)

    raw = DataFrame({
        "feature": transformed_names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    })
    raw["original"] = raw["feature"].map(feature_map)
    grouped = raw.groupby("original", as_index=False).agg({"importance": "sum", "std": "mean"})
    grouped.sort_values("importance", ascending=False, inplace=True)
    grouped.reset_index(drop=True, inplace=True)
    return grouped
