import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("💓 HEART STROKE PREDICTION")

st.write("Enter patient details below:")

# -------------------------
# Numeric Inputs
# -------------------------

age = st.number_input("Age", 1, 120, 30)
resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
cholesterol = st.number_input("Cholesterol", 0, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
max_hr = st.number_input("Max Heart Rate", 60, 250, 150)
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)

# -------------------------
# Categorical Inputs
# -------------------------

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

exercise_angina = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])

st_slope = st.selectbox(
    "ST Slope",
    ["Flat", "Up", "Down"]
)

# -------------------------
# Convert to Model Format
# -------------------------

input_data = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": 1 if fasting_bs == "Yes" else 0,
    "MaxHR": max_hr,
    "Oldpeak": oldpeak,
    "Sex_M": 1 if sex == "Male" else 0,
    "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
    "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
    "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0,
}

# Create dataframe
input_df = pd.DataFrame([input_data])

# Ensure correct column order
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# -------------------------
# Prediction Button
# -------------------------

if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
