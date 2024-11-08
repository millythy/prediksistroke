import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load scaler: {e}")

# Load the LSTM model
try:
    lstm_model = load_model('lstm_model.h5')
except Exception as e:
    st.error(f"Failed to load LSTM model: {e}")

# Load the SVM classifier
try:
    with open('svm_classifier.pkl', 'rb') as file:
        svm_classifier = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load SVM classifier: {e}")

# Define the prediction function
def make_prediction(input_data):
    # Preprocess input data
    input_scaled = scaler.transform(input_data)
    input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Extract features using the LSTM model
    lstm_features = lstm_model.predict(input_lstm)

    # Make prediction using the SVM model
    prediction = svm_classifier.predict(lstm_features)
    return prediction

# Streamlit app display
st.title("Stroke Prediction Application")

# Collect input from user
st.header("Enter Patient Information")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Encode categorical variables
gender = 1 if gender == "Male" else 0
ever_married = 1 if ever_married == "Yes" else 0
work_type_dict = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4}
work_type = work_type_dict[work_type]
residence_type = 1 if residence_type == "Urban" else 0
smoking_status_dict = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}
smoking_status = smoking_status_dict[smoking_status]

# Convert inputs to DataFrame format
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Predict button
if st.button("Predict"):
    prediction = make_prediction(input_data)
    if prediction[0] == 1:
        st.write("The patient is at risk of stroke.")
    else:
        st.write("The patient is not at risk of stroke.")