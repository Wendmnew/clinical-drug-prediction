import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Drug Type Prediction",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Drug Type Prediction using Machine Learning")
st.caption(
    "Educational mini-project based on the provided drug dataset. "
    "This tool is not a clinical decision-support system."
)

MODEL_PATHS = [
    Path("best_drug_prediction_pipeline.pkl"),
    Path("model/best_drug_prediction_pipeline.pkl"),
    Path("/content/drive/MyDrive/ML Project/best_drug_prediction_pipeline.pkl"),
]

@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "best_drug_prediction_pipeline.pkl was not found. "
        "Place the model in the project folder or update MODEL_PATHS."
    )

try:
    model = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        options=["Female", "Male"]
    )

    bp = st.selectbox(
        "Blood Pressure",
        options=["HIGH", "LOW", "NORMAL"]
    )

with col2:
    cholesterol = st.selectbox(
        "Cholesterol",
        options=["HIGH", "NORMAL"]
    )

    na_to_k = st.number_input(
        "Sodium to Potassium Ratio (Na_to_K)",
        min_value=1.0,
        max_value=50.0,
        value=15.0,
        step=0.1
    )

if st.button(
    "Predict Drug Type",
    type="primary",
    use_container_width=True
):
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "BP": bp,
        "Cholesterol": cholesterol,
        "Na_to_K": na_to_k
    }])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_names = model.classes_

    confidence = float(np.max(probabilities) * 100)

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Predicted Drug",
        str(prediction)
    )

    result_col2.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    probability_df = pd.DataFrame({
        "Drug Type": class_names,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    st.subheader("Prediction Probability")
    st.dataframe(
        probability_df.style.format({"Probability": "{:.2%}"}),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        probability_df.set_index("Drug Type")
    )

    st.info(
        "The probabilities shown are model probabilities for this educational "
        "machine-learning experiment and should not be interpreted as clinical certainty."
    )
