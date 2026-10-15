"""Streamlit demo on top of the model.

Run: streamlit run streamlit_app.py
It calls the model directly; to go through the API instead, replace the
prediction call with a requests.post to /predict.
"""
import streamlit as st

from app.model import churn_model

st.set_page_config(page_title="Customer Churn Predictor")
st.title("Customer Churn Predictor")
st.caption("Enter customer details to estimate the probability of leaving the bank.")

if not churn_model.is_loaded:
    churn_model.load()

col1, col2 = st.columns(2)
with col1:
    credit = st.slider("Credit score", 300, 900, 650)
    age = st.slider("Age", 18, 100, 42)
    tenure = st.slider("Tenure (years)", 0, 10, 3)
    balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
    salary = st.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0)
with col2:
    geo = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    products = st.slider("Number of products", 1, 4, 2)
    has_card = st.checkbox("Has credit card", value=True)
    active = st.checkbox("Is active member", value=True)

if st.button("Predict", type="primary"):
    rec = {
        "CreditScore": credit, "Geography": geo, "Gender": gender, "Age": age,
        "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
        "HasCrCard": int(has_card), "IsActiveMember": int(active), "EstimatedSalary": salary,
    }
    out = churn_model.predict([rec])[0]
    p = out["churn_probability"]
    st.metric("Churn probability", f"{p:.1%}")
    color = {"low": "green", "medium": "orange", "high": "red"}[out["risk_level"]]
    st.markdown(f"Risk level: **:{color}[{out['risk_level'].upper()}]**")
    st.progress(min(p, 1.0))
