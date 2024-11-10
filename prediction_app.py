import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load LSTM model
lstm_model = load_model('lstm_model.h5')

# Load SVM classifier
with open('svm_classifier.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

def make_prediction(input_data):
    # Ensure input_data is in 2D before creating a DataFrame
    input_data_2d = np.array(input_data).reshape(1, -1)
    input_df = pd.DataFrame(input_data_2d, columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
        'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ])

    # Transform data using preprocessor
    input_data_transformed = preprocessor.transform(input_df)
    print(f'Dimensi input setelah preprocessor: {input_data_transformed.shape}')
    
    # Reshape into 3D for LSTM
    input_lstm = input_data_transformed.reshape((1, 1, input_data_transformed.shape[1]))

    # Generate LSTM model features
    lstm_features = lstm_model.predict(input_lstm)

    # Combine LSTM features and scaled input
    final_input = np.concatenate([lstm_features, input_data_transformed], axis=1)

    # Final prediction from SVM
    prediction = svm_classifier.predict(final_input)
    return prediction[0]

# Streamlit UI
st.title('Prediksi Stroke')
st.write('Masukkan data untuk memprediksi kemungkinan terjadinya stroke.')

# User input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100)
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

# Show prediction when button is clicked
if st.button('Predict'):
    prediction = make_prediction(input_data)
    
    # Display prediction result
    if prediction > 0.5:
        st.write("Kemungkinan besar Anda mengalami stroke.")
    else:
        st.write("Kemungkinan besar Anda tidak mengalami stroke.")
