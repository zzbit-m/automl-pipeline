from typing import Any
from pandas import DataFrame
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import numpy.typing as npt


def _classification_models() -> dict[str, BaseEstimator]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_jobs=-1),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "XGBoost": XGBClassifier(n_jobs=-1, verbosity=0),
        "LightGBM": LGBMClassifier(n_jobs=-1, verbose=-1),
    }


def _regression_models() -> dict[str, BaseEstimator]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_jobs=-1),
        "XGBoost": XGBRegressor(n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_jobs=-1, verbose=-1),
    }


def _encode_target(y: npt.NDArray) -> tuple[npt.NDArray, LabelEncoder | None]:
    if y.dtype.kind in ("U", "O", "S"):
        le = LabelEncoder()
        return le.fit_transform(y), le
    return y, None


def compare_models(
    X: npt.NDArray, y: npt.NDArray, problem_type: str, cv: int = 5, test_size: float = 0.2
) -> tuple[dict[str, Any], BaseEstimator, float]:
    X = np.asarray(X)
    y = np.asarray(y)

    if y.dtype.kind in ("U", "O", "S"):
        y, _ = _encode_target(y)

    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=42
    )

    models = _classification_models() if problem_type == "classification" else _regression_models()
    scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"

    results: dict[str, Any] = {}
    best_cv_score = -np.inf
    best_model: BaseEstimator | None = None

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = scores.mean()
        results[name] = {
            "mean": mean_score,
            "std": scores.std(),
            "scores": scores.tolist(),
        }
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_model = model

    if best_model is None:
        raise RuntimeError("no models trained")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    if problem_type == "classification":
        holdout_score = accuracy_score(y_test, y_pred)
    else:
        holdout_score = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    best_model.fit(X, y)

    return results, best_model, holdout_score


def leaderboard(results: dict[str, Any]) -> DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({"model": name, "score": r["mean"], "std": r["std"]})
    df = DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
