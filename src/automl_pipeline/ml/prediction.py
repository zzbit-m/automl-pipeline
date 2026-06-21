from pandas import DataFrame
from sklearn.base import BaseEstimator

from automl_pipeline.ml.preprocessing import Preprocessor


def predict(
    model: BaseEstimator,
    preprocessor: Preprocessor,
    df: DataFrame,
    target: str,
) -> DataFrame:
    X_new = preprocessor.transform(df, target)
    preds = model.predict(X_new)

    result = df.copy()
    result["prediction"] = preds

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        proba = model.predict_proba(X_new)
        for i, cls in enumerate(model.classes_):
            result[f"prob_{cls}"] = proba[:, i]

    return result
