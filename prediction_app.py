import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load LSTM model
lstm_model = load_model("lstm_model.h5")

# Load SVM classifier
with open('svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Fungsi untuk membuat prediksi
def make_prediction(input_data):
    # Preprocess input data
    input_scaled = scaler.transform(input_data)
    input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Ekstrak fitur menggunakan model LSTM
    lstm_features = lstm_model.predict(input_lstm)

    # Prediksi menggunakan SVM
    predictions = svm_classifier.predict(lstm_features)
    return predictions

# Tampilan Streamlit
st.title("Stroke Prediction Application")

# Input data dari pengguna
st.header("Masukkan Informasi Pasien")
gender = st.selectbox("Gender", ["Male", "Female"])
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Konversi input ke bentuk DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Tombol untuk melakukan prediksi
if st.button("Predict"):
    # Lakukan prediksi
    prediction = make_prediction(input_data)
    if prediction[0] == 1:
        st.write("Pasien memiliki risiko terkena stroke.")
    else:
        st.write("Pasien tidak memiliki risiko terkena stroke.")
