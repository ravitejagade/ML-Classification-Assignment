
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Classification Dashboard")
st.write("BITS Pilani | M.Tech AIML | Assignment 2")

MODEL_PATH = "models"

models = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Gaussian Naive Bayes": "Gaussian_Naive_Bayes.pkl",
    "Support Vector Machine": "SVM.pkl"
}

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose Classification Model",
    list(models.keys())
)

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    if isinstance(feature_columns, dict):
        feature_columns = feature_columns.get(
            "feature_columns",
            feature_columns.get("columns", list(feature_columns))
        )

    target_column = "target"

    if target_column not in df.columns:
        st.error("Target column 'target' is missing.")
        st.stop()

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        st.error(f"Missing features: {missing}")
        st.stop()

    X = df[feature_columns]
    y = df[target_column]

    model_file = os.path.join(
        MODEL_PATH,
        models[selected_model]
    )

    model = joblib.load(model_file)

    predictions = model.predict(X)

    st.subheader("Model Evaluation Results")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probabilities)

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        auc = roc_auc_score(y, scores)

    else:
        auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions)
    }

    cols = st.columns(3)

    for i, (name, value) in enumerate(metrics.items()):
        cols[i % 3].metric(
            name,
            f"{value:.4f}" if pd.notna(value) else "N/A"
        )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
    )

    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.success("Model evaluation completed successfully!")

else:
    st.info("Please upload your test_data.csv file.")