import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ------------------------------------------------------
# Load model and scaler
# ------------------------------------------------------
model_path = "heart-disease-model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found! Please ensure 'heart-disease-model.pkl' is in the same folder as app.py.")
    st.stop()

with open(model_path, "rb") as file:
    model, scaler = pickle.load(file)

# ------------------------------------------------------
# App configuration
# ------------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon="🩺", layout="centered")

st.title("🩺 Heart Disease Prediction App")
st.write("This app predicts the likelihood of heart disease based on patient health data.")

st.markdown("---")

# ------------------------------------------------------
# Define columns (must match training exactly)
# ------------------------------------------------------
categorical_cols = [
    'Gender', 'ChestPainType', 'FastingBS', 'RestingECG',
    'ExerciseAngina', 'ST_Slope', 'MajorVessels', 'Thalassemia'
]
numerical_cols = ['Age', 'Cholesterol', 'RestingBp', 'MaxHR', 'ST_Depress']

# ------------------------------------------------------
# Collect user inputs
# ------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 20, 100, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", [
        "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    RestingBp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (≤120 mg/dl)", "Yes (>120 mg/dl)"])
    RestingECG = st.selectbox("Resting ECG Results", [
        "Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    ST_Depress = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", ["Downsloping", "Flat", "Upsloping"])
    MajorVessels = st.selectbox("Major Vessels Colored by Fluoroscopy", [
        "0 vessels", "1 vessel", "2 vessels", "3 vessels"])
    Thalassemia = st.selectbox("Thalassemia", ["Fixed Defect", "Normal", "Reversible Defect"])

# ------------------------------------------------------
# Map labels back to numeric values
# ------------------------------------------------------
Gender = 1 if Gender == "Male" else 0
ChestPainType = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(ChestPainType)
FastingBS = 0 if FastingBS == "No (≤120 mg/dl)" else 1
RestingECG = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(RestingECG)
ExerciseAngina = 0 if ExerciseAngina == "No" else 1
ST_Slope = ["Downsloping", "Flat", "Upsloping"].index(ST_Slope)
MajorVessels = ["0 vessels", "1 vessel", "2 vessels", "3 vessels"].index(MajorVessels)
Thalassemia = ["Fixed Defect", "Normal", "Reversible Defect"].index(Thalassemia) + 1

input_dict = {
    'Age': Age,
    'Gender': Gender,
    'ChestPainType': ChestPainType,
    'RestingBp': RestingBp,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'RestingECG': RestingECG,
    'MaxHR': MaxHR,
    'ExerciseAngina': ExerciseAngina,
    'ST_Depress': ST_Depress,
    'ST_Slope': ST_Slope,
    'MajorVessels': MajorVessels,
    'Thalassemia': Thalassemia
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]
    risk_pct = int(proba[1] * 100)

    if prediction == 1:
        st.error(f"⚠️ High risk of Heart Disease detected! ({risk_pct}% probability)\nPlease consult a cardiologist.")
    else:
        st.success(f"✅ No signs of Heart Disease detected. ({risk_pct}% risk probability)")
