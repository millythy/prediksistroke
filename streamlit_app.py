import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Define constants
model_path = os.path.join(os.getcwd(), 'lstm_model.h5')
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
threshold = 0.5

# Load model and scaler
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    st.success("Scaler loaded successfully!")
except (FileNotFoundError, pickle.UnpicklingError) as e:
    st.error(f"Error loading scaler: {e}")
    scaler = None

# Define prediction function
def predict_stroke(features: pd.DataFrame) -> int:
    if scaler is None:
        st.error("Scaler is not available.")
        return None
    if model is None:
        st.error("Model is not available.")
        return None
    
    # Transform and reshape data features
    features_scaled = scaler.transform(features)
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))
    prediction = model.predict(features_scaled)
    return (prediction > threshold).astype("int32")

# Streamlit interface
st.title("Stroke Prediction Classifier")
st.write("Masukkan fitur yang akan dianalisis:")

# Input features
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=0)
hypertension = st.selectbox("Hypertension", options=[0, 1])
heart_disease = st.selectbox("Heart Disease", options=[0, 1])
ever_married = st.selectbox("Ever Married", options=["Yes", "No"])
work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level")
bmi = st.number_input("BMI")
smoking_status = st.selectbox("Smoking Status", options=["formerly smoked", "never smoked", "smokes", "Unknown"])

# Map input text to numeric values for model
input_data = pd.DataFrame([[
    1 if gender == "Male" else 0,
    age,
    hypertension,
    heart_disease,
    1 if ever_married == "Yes" else 0,
    {"Private": 1, "Self-employed": 2, "Govt_job": 3, "Children": 4, "Never_worked": 5}[work_type],
    1 if residence_type == "Urban" else 0,
    avg_glucose_level,
    bmi,
    {"formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0}[smoking_status]
]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

# Check for valid input
if age < 0 or avg_glucose_level <= 0 or bmi <= 0:
    st.error("Please provide valid input for age, glucose level, and BMI.")

# Prediction button
if st.button("Prediksi"):
    prediction = predict_stroke(input_data)
    if prediction is not None:
        st.write("Hasil Prediksi: ", "Stroke" if prediction[0][0] == 1 else "Tidak Stroke")