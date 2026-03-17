import streamlit as st
import joblib
import pandas as pd
import os

last_name = "Mugimba"

# Absolute path to model
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(repo_root, 'models', f'{last_name}_best_model.pkl')

# Load model
try:
    model = joblib.load(model_path)
    st.success(f"Model loaded from {model_path}!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Trained features (exact from your model)
trained_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']

st.title("Mugimba Diabetes Risk Predictor - Patient Intelligence")

# Inputs only for trained features
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=100, value=30)

if st.button("Predict"):
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    input_df = pd.DataFrame([input_data])[trained_features]  # Exact order
    pred = model.predict(input_df)[0]
    risk = "High Risk (Diabetes Likely)" if pred == 1 else "Low Risk (No Diabetes)"
    st.write(f"Predicted Risk: {risk}")