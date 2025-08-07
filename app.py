import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
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

# Create input array
input_data = np.array([[gender_code, married_code, dependents_code, education_code,
                        self_employed_code, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history_code, property_area_code]])

# Scale numeric features
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]  # Probability of default

    # Gauge Chart for Risk
    st.subheader("📉 Risk Probability Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 2),
        title={'text': "Probability of Default (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if prediction[0] == 1 else "green"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgreen"},
                   {'range': [50, 75], 'color': "yellow"},
                   {'range': [75, 100], 'color': "red"}
               ]}
    ))
    st.plotly_chart(fig_gauge)

    if prediction[0] == 1:
        st.error("❌ High Risk: Loan Likely to Default.")
    else:
        st.success("✅ Low Risk: Loan Likely to be Approved.")

# -----------------------------
# 📊 Additional Visualizations
# -----------------------------

st.markdown("---")
st.header("📊 Visual Insights Based on Your Inputs")

# 1. Bar Chart of Applicant vs Coapplicant Income
st.subheader("1. Applicant vs Coapplicant Income")
fig1, ax1 = plt.subplots()
sns.barplot(x=["Applicant", "Coapplicant"], y=[applicant_income, coapplicant_income], ax=ax1)
ax1.set_ylabel("Income")
st.pyplot(fig1)

# 2. Pie Chart: Property Area Distribution (Fixed Example)
st.subheader("2. Sample Loan Distribution by Property Area")
area_counts = pd.Series({"Urban": 45, "Semiurban": 35, "Rural": 20})
fig2, ax2 = plt.subplots()
ax2.pie(area_counts, labels=area_counts.index, autopct='%1.1f%%', startangle=90)
ax2.axis('equal')
st.pyplot(fig2)

# 3. Histogram: Loan Amount Range
st.subheader("3. Loan Amount Distribution (Sample)")
sample_loans = np.random.normal(loc=loan_amount if loan_amount > 0 else 150, scale=50, size=100)
fig3, ax3 = plt.subplots()
sns.histplot(sample_loans, bins=20, kde=True, ax=ax3)
ax3.set_xlabel("Loan Amount (in thousands)")
st.pyplot(fig3)

# 4. Scatter Plot: Income vs Loan Amount (Single Point + Sample)
st.subheader("4. Income vs Loan Amount")
sample_income = np.random.normal(loc=applicant_income, scale=3000, size=50)
sample_loan = np.random.normal(loc=loan_amount if loan_amount > 0 else 150, scale=50, size=50)
fig4, ax4 = plt.subplots()
ax4.scatter(sample_income, sample_loan, alpha=0.4, label="Sample")
ax4.scatter(applicant_income, loan_amount, color='red', s=100, label="Your Entry")
ax4.set_xlabel("Applicant Income")
ax4.set_ylabel("Loan Amount")
ax4.legend()
st.pyplot(fig4)
