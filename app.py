import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64

# -------------------------
# Background Image Function
# -------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("bank.png")

# -------------------------
# Load model and scaler
# -------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# -------------------------
# Input fields
# -------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.slider("Loan Amount (in thousands)", min_value=0, max_value=500, value=150, step=1)
loan_amount_term = st.slider("Loan Amount Term (in months)", min_value=12, max_value=480, value=360, step=12)
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs
gender_code = 1 if gender == "Male" else 0
married_code = 1 if married == "Yes" else 0
dependents_code = 3 if dependents == "3+" else int(dependents)
education_code = 0 if education == "Graduate" else 1
self_employed_code = 1 if self_employed == "Yes" else 0
credit_history_code = 1 if credit_history == "Good (1)" else 0
property_area_code = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

input_data = np.array([[gender_code, married_code, dependents_code, education_code,
                        self_employed_code, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history_code, property_area_code]])

input_scaled = scaler.transform(input_data)

# PDF function
def create_pdf(data_dict, filename="loan_prediction.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "Loan Prediction Report")
    c.line(50, 745, 550, 745)
    
    y = 720
    for key, value in data_dict.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20
    
    c.save()

# Predict & Download
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    prediction_result = "✅ Low Risk: Loan Likely to be Approved." if prediction[0] == 0 else "❌ High Risk: Loan Likely to Default."
    
    if prediction[0] == 0:
        st.success(prediction_result)
    else:
        st.error(prediction_result)
    
    user_data = {
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
    
    create_pdf(user_data)
    with open("loan_prediction.pdf", "rb") as pdf_file:
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_file,
            file_name="loan_prediction.pdf",
            mime="application/pdf"
        )

# -------------------------
# Visuals
# -------------------------
st.subheader("📊 Data Insights")
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(x=["Applicant", "Coapplicant"], y=[applicant_income, coapplicant_income], ax=ax)
    ax.set_title("Income Comparison")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot([loan_amount], kde=False, ax=ax)
    ax.set_title("Loan Amount Distribution")
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    sns.boxplot(y=[loan_amount_term], ax=ax)
    ax.set_title("Loan Term Spread")
    st.pyplot(fig)

with col4:
    fig = px.pie(
        names=["Dependents: " + str(dependents), "Others"],
        values=[1, 3],
        title="Dependents Share"
    )
    st.plotly_chart(fig)








