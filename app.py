import streamlit as st
import pandas as pd
import numpy as np
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Default Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.slider("Loan Amount (in thousands)", min_value=0, max_value=500, value=150)
loan_amount_term = st.slider("Loan Amount Term (in months)", min_value=12, max_value=480, value=360)
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to numeric codes
gender_code = 1 if gender == "Male" else 0
married_code = 1 if married == "Yes" else 0
dependents_code = 3 if dependents == "3+" else int(dependents)
education_code = 0 if education == "Graduate" else 1
self_employed_code = 1 if self_employed == "Yes" else 0
credit_history_code = 1 if credit_history == "Good (1)" else 0
property_area_code = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Prepare data
input_data = np.array([[gender_code, married_code, dependents_code, education_code,
                        self_employed_code, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history_code, property_area_code]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
prediction_result = "Low Risk: Loan Likely to be Approved" if prediction[0] == 0 else "High Risk: Loan Likely to Default"

# Function to create PDF in memory
def create_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "Loan Prediction Report")
    c.line(50, 745, 550, 745)

    y = 720
    info = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self Employed": self_employed,
        "Applicant Income": applicant_income,
        "Coapplicant Income": coapplicant_income,
        "Loan Amount (000s)": loan_amount,
        "Loan Amount Term (months)": loan_amount_term,
        "Credit History": "Good" if credit_history_code == 1 else "Bad",
        "Property Area": property_area,
        "Prediction": prediction_result
    }
    for key, value in info.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()
    buffer.seek(0)
    return buffer

# Single button: generate & download PDF
st.download_button(
    label="📥 Download Loan Report as PDF",
    data=create_pdf(),
    file_name="loan_prediction.pdf",
    mime="application/pdf"
)















