import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st
import pandas as pd
from pandas import DataFrame
import altair as alt
import joblib

from automl_pipeline.ml.ingestion import (
    load_csv, validate_target, validate_inference_columns, IngestionError,
)
from automl_pipeline.ml.summary import summarize
from automl_pipeline.ml.detection import detect_problem
from automl_pipeline.ml.preprocessing import Preprocessor
from automl_pipeline.ml.training import compare_models, leaderboard
from automl_pipeline.ml.importance import compute_importance
from automl_pipeline.ml.prediction import predict
from automl_pipeline.storage.database import init_db, insert_job, insert_model_result
from automl_pipeline.storage.database import insert_predictions, get_jobs, get_best_model
from automl_pipeline.storage.database import get_predictions, get_model_results

st.set_page_config(page_title="AutoML Pipeline", layout="wide")
init_db()

for key in ("df", "trained", "job_id", "view_job_id"):
    if key not in st.session_state:
        st.session_state[key] = None

st.session_state.setdefault("df", None)
st.session_state.setdefault("trained", False)
st.session_state.setdefault("job_id", None)
st.session_state.setdefault("view_job_id", None)

st.title("AutoML Pipeline")
st.markdown("Upload a CSV, pick the target column, and let the pipeline find the best model.")

uploaded = st.file_uploader("Upload your CSV", type="csv")

if uploaded:
    df = load_csv(uploaded)
    st.session_state.df = df

    with st.expander("Preview data", expanded=True):
        st.dataframe(df.head())

    with st.expander("Data summary"):
        info = summarize(df)
        ov = info["overview"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{ov['rows']:,}")
        col2.metric("Columns", ov["columns"])
        col3.metric("Missing cells", f'{ov["missing_pct"]}%')
        col4.metric("Memory", f'{ov["memory_kb"]} KB')

        stats_df = DataFrame(info["columns"])
        display_cols = ["name", "dtype", "non_null", "null_pct", "unique"]
        if "mean" in stats_df.columns:
            display_cols += ["mean", "std", "min", "max"]
        if "top" in stats_df.columns:
            display_cols += ["top"]
        st.dataframe(
            stats_df[display_cols].style.format(
                {"null_pct": "{:.1f}%", "mean": "{:.3f}", "std": "{:.3f}",
                 "min": "{:.3f}", "max": "{:.3f}"}
            ),
            use_container_width=True,
        )

    target = st.selectbox("Select the target column to predict", df.columns)

    with st.expander("Target distribution"):
        vals = df[target].value_counts().reset_index()
        vals.columns = [target, "count"]
        st.bar_chart(vals.set_index(target))

    if st.button("Train models", type="primary"):
        progress = st.progress(0, text="Validating...")
        try:
            validate_target(df, target)

            progress.progress(10, text="Detecting problem type...")
            problem = detect_problem(df, target)
            st.info(f"Detected problem type: **{problem}**")

            progress.progress(20, text="Preprocessing data...")
            preprocessor = Preprocessor()
            X = preprocessor.fit_transform(df, target)
            y = df[target].values

            progress.progress(30, text="Training models with cross-validation...")
            results, best_model, holdout_score = compare_models(X, y, problem)

            progress.progress(70, text="Building leaderboard and saving results...")
            job_id = insert_job(uploaded.name, target, problem, holdout_score)
            st.session_state.job_id = job_id

            lb = leaderboard(results)
            for _, row in lb.iterrows():
                insert_model_result(job_id, row["model"], row["score"], row["std"])

            progress.progress(80, text="Computing feature importance...")
            importance_df = compute_importance(best_model, X, y, preprocessor, problem)

            progress.progress(90, text="Done!")
            st.session_state.trained = True
            st.session_state.results = results
            st.session_state.best_model = best_model
            st.session_state.preprocessor = preprocessor
            st.session_state.problem = problem
            st.session_state.target = target
            st.session_state.df_full = df
            st.session_state.holdout_score = holdout_score
            st.session_state.importance_df = importance_df

            progress.empty()

        except Exception as e:
            progress.empty()
            st.error(f"Training failed: {e}")

if st.session_state.trained:
    st.subheader("Leaderboard")
    lb = leaderboard(st.session_state.results)
    st.dataframe(
        lb.style.format({"score": "{:.4f}", "std": "{:.4f}"}),
        use_container_width=True,
    )

    best_row = lb.iloc[0]
    metric_label = "Accuracy" if st.session_state.problem == "classification" else "RMSE"
    holdout_label = f"Holdout {metric_label}: {st.session_state.holdout_score:.4f}"
    st.success(
        f"Best model: **{best_row['model']}** — "
        f"CV Score: {best_row['score']:.4f} — {holdout_label}"
    )

    with st.expander("Feature importance"):
        imp_df = st.session_state.importance_df
        top_n = min(10, len(imp_df))
        chart_df = imp_df.head(top_n).iloc[::-1]
        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("original:N", title=None, sort="-x"),
                color=alt.condition(
                    alt.datum.importance > 0,
                    alt.value("#1f77b4"),
                    alt.value("#ff7f0e"),
                ),
            ).properties(height=30 * top_n),
            use_container_width=True,
        )

    model_bytes = joblib.dumps({
        "model": st.session_state.best_model,
        "preprocessor": st.session_state.preprocessor,
        "feature_columns": st.session_state.preprocessor.feature_columns,
    })
    st.download_button(
        "Download best model (.pkl)",
        data=model_bytes,
        file_name=f"best_model_{best_row['model'].replace(' ', '_')}.pkl",
        mime="application/octet-stream",
    )

    st.subheader("Run predictions on new data")
    infer_file = st.file_uploader("Upload CSV to predict", type="csv", key="infer")

    if infer_file:
        infer_df = load_csv(infer_file)
        st.dataframe(infer_df.head())

        if st.button("Run predictions", type="primary"):
            try:
                validate_inference_columns(
                    infer_df,
                    st.session_state.preprocessor.feature_columns,
                    st.session_state.target,
                )

                preds = predict(
                    st.session_state.best_model,
                    st.session_state.preprocessor,
                    infer_df,
                    st.session_state.target,
                )

                job_id = st.session_state.job_id
                assert job_id is not None
                prediction_rows = []
                for idx, row in preds.iterrows():
                    prediction_rows.append({
                        "row_index": int(idx),
                        "prediction": float(row["prediction"]),
                    })
                insert_predictions(job_id, prediction_rows)

                st.dataframe(preds, use_container_width=True)

                csv_data = preds.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions (CSV)",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except IngestionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if st.session_state.view_job_id is not None:
    st.subheader("Past predictions")
    view_preds = get_predictions(st.session_state.view_job_id)
    if view_preds:
        pred_df = DataFrame(view_preds)
        st.dataframe(pred_df, use_container_width=True)
        csv_data = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions (CSV)",
            data=csv_data,
            file_name=f"predictions_{st.session_state.view_job_id}.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions recorded for this run.")

    model_rows = get_model_results(st.session_state.view_job_id)
    if model_rows:
        st.caption("Model scores from this run:")
        st.dataframe(
            DataFrame(model_rows).style.format({"score": "{:.4f}", "std": "{:.4f}"}),
            use_container_width=True,
        )

with st.sidebar:
    st.subheader("Run history")
    jobs = get_jobs(5)
    for j in jobs:
        best = get_best_model(j["id"])
        label = f"**{j['filename']}** — {j['problem_type']}"
        if best:
            label += f" — Best: {best['model_name']} ({best['score']:.3f})"
        if j.get("holdout_score"):
            label += f" [Holdout: {j['holdout_score']:.3f}]"
        cols = st.columns([3, 1])
        cols[0].caption(label)
        if cols[1].button("View", key=f"view_{j['id']}"):
            st.session_state.view_job_id = j["id"]
            st.rerun()
