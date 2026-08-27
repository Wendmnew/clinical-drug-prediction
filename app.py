import streamlit as st
import pandas as pd
import numpy as np
import dill
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="ML Drug Classifier", page_icon="💊", layout="centered")

# ============================================================
# APPLICATION TITLE
# ============================================================
st.title("💊 Drug Type Prediction using Machine Learning")
st.markdown("Enter the patient's demographic and clinical information below.")
st.divider()

# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================
@st.cache_resource
def load_model_artifact():
    MODEL_PATH = "/content/gdrive/MyDrive/ML-Projects/Drug_Type_Prediction/best_drug_prediction_pipeline.pkl"
    LOCAL_PATH = "best_drug_prediction_pipeline.pkl"
    
    if os.path.exists(MODEL_PATH):
        path = MODEL_PATH
    elif os.path.exists(LOCAL_PATH):
        path = LOCAL_PATH
    else:
        st.error("❌ Model file not found!")
        st.stop()
        
    try:
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception as exc:
        st.error(f"❌ Failed to load model: {exc}")
        st.stop()

artifact = load_model_artifact()

# ============================================================
# LOAD MODEL AND LABEL ENCODER
# ============================================================
if isinstance(artifact, dict):
    model = artifact.get("model")
    label_encoder = artifact.get("label_encoder")
    if model is None:
        st.error("❌ Saved artifact does not contain the model.")
        st.stop()
else:
    model = artifact
    label_encoder = None

# ============================================================
# INPUT SECTION
# ============================================================
st.subheader("👤 Patient Information")
col1, col2 = st.columns(2)

with col1:
    # Minimum age set to 15 (dataset bound), Maximum age set to 74 (dataset bound)
    age = st.number_input("Age (Min: 15, Max: 74)", min_value=15, max_value=74, value=30, step=1)
    sex = st.selectbox("Sex", options=["F", "M"])
    bp = st.selectbox("Blood Pressure (BP)", options=["HIGH", "LOW", "NORMAL"])

with col2:
    na_to_k = st.number_input("Sodium to Potassium Ratio (Na_to_K)", min_value=1.0, max_value=50.0, value=15.0, step=0.1)
    cholesterol = st.selectbox("Cholesterol", options=["HIGH", "NORMAL"])

# ============================================================
# CREATE INPUT DATA
# ============================================================
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "BP": bp,
    "Cholesterol": cholesterol,
    "Na_to_K": na_to_k
}])

# ============================================================
# PREDICTION
# ============================================================
st.markdown("<style>div.stButton > button {background-color: #4CAF50 !important; color: white !important;}</style>", unsafe_allow_html=True)
if st.button("Predict Optimal Drug", type="primary", use_container_width=True):
    try:
        # Prediction
        raw_prediction = model.predict(input_data)[0]

        # Prediction probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
        else:
            probabilities = None

        # Decode prediction
        if label_encoder is not None:
            predicted_drug = label_encoder.inverse_transform([int(raw_prediction)])[0]
            class_names = label_encoder.classes_
        else:
            predicted_drug = str(raw_prediction)
            if hasattr(model, "classes_"):
                class_names = model.classes_
            else:
                class_names = ["DrugY", "drugA", "drugB", "drugC", "drugX"]

        # Confidence
        if probabilities is not None:
            confidence = float(np.max(probabilities)) * 100
        else:
            confidence = None

        # ====================================================
        # RESULT
        # ====================================================
        st.divider()
        st.subheader("🧪 Prediction Result")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Predicted Drug", str(predicted_drug))

        with res_col2:
            if confidence is not None:
                st.metric("Confidence", f"{confidence:.2f}%")
            else:
                st.metric("Confidence", "N/A")

        # ====================================================
        # PROBABILITIES
        # ====================================================
        if probabilities is not None:
            probability_values = np.asarray(probabilities, dtype=float)

            if len(class_names) == len(probability_values):
                prob_df = pd.DataFrame({
                    "Drug Type": class_names,
                    "Probability": probability_values
                })
                
                prob_df["Percentage"] = prob_df["Probability"] * 100
                prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

                # Probability chart
                st.subheader("📊 Prediction Probability")
                chart_df = prob_df.set_index("Drug Type")[["Percentage"]]
                st.bar_chart(chart_df)

                # Highest probability caption
                st.caption(f"Highest probability: {predicted_drug} — {confidence:.2f}%")

    except Exception as exc:
        st.error(f"❌ Prediction failed: {exc}")
