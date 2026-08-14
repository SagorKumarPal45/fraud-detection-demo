"""Streamlit demo app for credit card fraud detection."""

import os
import sys

# Streamlit must launch the app — `python app.py` alone won't open the website.
if __name__ == "__main__" and os.environ.get("STREAMLIT_RUN_MAIN") != "true":
    import subprocess

    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
    sys.exit(0)

from pathlib import Path

import pandas as pd
import streamlit as st

from predict import AVAILABLE_MODELS, UNAVAILABLE_MODELS, predict_fraud, summarize_results

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("Credit Card Fraud Detection Demo")
st.markdown(
    """
    Upload a CSV file in the same format as **creditcard.csv** (`Time`, `V1`–`V28`, `Amount`).
    The model will classify each transaction as **Normal** or **Fraud**.

    **Group 4 – Threat Detection Project**
    """
)

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Choose model", list(AVAILABLE_MODELS.keys()))
    st.markdown("---")
    st.subheader("Download sample data")
    sample_upload = BASE_DIR / "sample_upload_data.csv"
    sample_labeled = BASE_DIR / "sample_test_data.csv"
    if sample_upload.exists():
        st.download_button(
            "Sample CSV (no labels)",
            data=sample_upload.read_bytes(),
            file_name="sample_upload_data.csv",
            mime="text/csv",
        )
    if sample_labeled.exists():
        st.download_button(
            "Sample CSV (with Class labels)",
            data=sample_labeled.read_bytes(),
            file_name="sample_test_data.csv",
            mime="text/csv",
        )
    st.markdown("---")
    st.caption("Unavailable models (corrupted files):")
    for name in UNAVAILABLE_MODELS:
        st.caption(f"• {name}")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded **{len(df):,}** rows × **{len(df.columns)}** columns")

        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run fraud detection", type="primary"):
            with st.spinner(f"Running {model_name}..."):
                results = predict_fraud(df, model_name)
                summary = summarize_results(results)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total transactions", summary["total_transactions"])
            col2.metric("Flagged as fraud", summary["flagged_fraud"])
            col3.metric("Flagged as normal", summary["flagged_normal"])
            col4.metric("Avg fraud probability", f"{summary['avg_fraud_probability']:.2%}")

            if "accuracy" in summary:
                st.markdown("### Evaluation (Class column detected)")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Actual fraud cases", summary["actual_fraud"])
                e2.metric("Actual normal cases", summary["actual_normal"])
                e3.metric("Accuracy", f"{summary['accuracy']:.2%}")
                if "fraud_recall" in summary:
                    e4.metric("Fraud recall", f"{summary['fraud_recall']:.2%}")

            st.markdown("### Predictions")
            st.dataframe(results, use_container_width=True)

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Error: {exc}")
else:
    st.info("Upload a CSV file or download the sample data from the sidebar to get started.")

    st.markdown(
        """
        ### How it works
        1. **RobustScaler** scales `Time` and `Amount`
        2. **Feature selection** keeps the 15 most correlated V-features
        3. **PCA** reduces to 6 components
        4. Selected ML model predicts fraud (0 = normal, 1 = fraud)

        ### Expected CSV columns
        `Time`, `V1`, `V2`, … `V28`, `Amount` — optional `Class` for accuracy check
        """
    )
