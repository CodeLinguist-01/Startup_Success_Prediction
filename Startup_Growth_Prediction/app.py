import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import os

# --- Load model and scaler ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLS_PATH = os.path.join(BASE_DIR, "input_columns.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(COLS_PATH, "rb") as f:
        input_columns = joblib.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- Streamlit App ---
st.set_page_config(page_title="Startup Success Predictor", layout="centered")
st.title("🚀 Startup Growth Success Predictor")
st.write("Enter your startup details to predict its success category.")

# --- Input Fields ---
funding = st.number_input("Total Funding ($M)", 0.0, 1000.0, 50.0)
employees = st.number_input("Number of Employees", 0, 10000, 100)
revenue = st.number_input("Annual Revenue ($M)", 0.0, 1000.0, 10.0)
valuation = st.number_input("Valuation ($B)", 0.0, 100.0, 1.0)
followers = st.number_input("Social Media Followers", 0, 10000000, 50000)
tech_stack = st.slider("Tech Stack Count", 1, 10, 3)
acquired = 1 if st.selectbox("Acquired?", ["No", "Yes"]) == "Yes" else 0
ipo = 1 if st.selectbox("IPO?", ["No", "Yes"]) == "Yes" else 0
founded_year = st.number_input("Founded Year", 1980, 2025, 2015)
customer_base = st.number_input("Customer Base (Millions)", 0.0, 1000.0, 1.0)

# --- One-hot encoded categorical values ---
# Update these options based on your model's training data
country_options = [col.replace("Country_", "") for col in input_columns if col.startswith("Country_")]
industry_options = [col.replace("Industry_", "") for col in input_columns if col.startswith("Industry_")]
stage_options = [col.replace("Funding Stage_", "") for col in input_columns if col.startswith("Funding Stage_")]

selected_country = st.selectbox("Country", country_options)
selected_industry = st.selectbox("Industry", industry_options)
selected_stage = st.selectbox("Funding Stage", stage_options)

# --- Prepare input vector ---
input_data = {
    'Total Funding ($M)': funding,
    'Number of Employees': employees,
    'Annual Revenue ($M)': revenue,
    'Valuation ($B)': valuation,
    'Social Media Followers': followers,
    'Tech Stack Count': tech_stack,
    'Acquired?': acquired,
    'IPO?': ipo,
    'Founded Year': founded_year,
    'Customer Base (Millions)': customer_base,
}

# Initialize all columns to 0
input_df = pd.DataFrame([np.zeros(len(input_columns))], columns=input_columns)

# Assign numerical values
for col in input_data:
    if col in input_df.columns:
        input_df[col] = input_data[col]

# Set one-hot encoded values
input_df[f"Country_{selected_country}"] = 1
input_df[f"Industry_{selected_industry}"] = 1
input_df[f"Funding Stage_{selected_stage}"] = 1

# --- Predict ---
if st.button("Predict Success"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        categories = {0: "Low", 1: "Medium", 2: "High"}
        st.success(f"🎯 Predicted Success Category: **{categories.get(prediction, 'Unknown')}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
