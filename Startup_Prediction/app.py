import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# Load model and supporting files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
input_columns = joblib.load(open("input_columns.pkl", "rb"))

st.set_page_config(page_title="Startup Growth Tracker", page_icon="🚀")
st.title("🚀 Startup Growth Success Score Predictor")

st.markdown("Enter the details of your startup to predict its success potential.")

# Basic Inputs
funding = st.number_input("Total Funding ($M)", 0, 1000, 50)
employees = st.number_input("Number of Employees", 0, 10000, 100)
revenue = st.number_input("Annual Revenue ($M)", 0, 1000, 10)
valuation = st.number_input("Valuation ($B)", 0.0, 100.0, 1.0)
followers = st.number_input("Social Media Followers", 0, 10000000, 50000)
tech_stack = st.slider("Tech Stack Count", 1, 10, 3)
acquired = 1 if st.selectbox("Acquired?", ["No", "Yes"]) == "Yes" else 0
ipo = 1 if st.selectbox("IPO?", ["No", "Yes"]) == "Yes" else 0
founded_year = st.number_input("Founded Year", 1980, 2025, 2015)
customer_base = st.number_input("Customer Base (Millions)", 0.0, 1000.0, 1.0)

# Categorical selections
country = st.selectbox("Country", ['Brazil', 'Canada', 'China', 'France', 'Germany', 'India', 'Japan', 'UK', 'USA'])
industry = st.selectbox("Industry", ['E-commerce', 'EdTech', 'Energy', 'FinTech', 'FoodTech', 'Gaming', 'Healthcare', 'Logistics', 'Tech'])
funding_stage = st.selectbox("Funding Stage", ['Seed', 'Series A', 'Series B', 'Series C'])

# Build input dictionary with zeros
input_data = dict.fromkeys(input_columns, 0)

# Assign numerical inputs
input_data['Founded Year'] = founded_year
input_data['Total Funding ($M)'] = funding
input_data['Number of Employees'] = employees
input_data['Annual Revenue ($M)'] = revenue
input_data['Valuation ($B)'] = valuation
input_data['Acquired?'] = acquired
input_data['IPO?'] = ipo
input_data['Customer Base (Millions)'] = customer_base
input_data['Social Media Followers'] = followers
input_data['Tech Stack Count'] = tech_stack

# Assign one-hot categorical features
if f'Country_{country}' in input_data:
    input_data[f'Country_{country}'] = 1

if f'Industry_{industry}' in input_data:
    input_data[f'Industry_{industry}'] = 1

if f'Funding Stage_{funding_stage}' in input_data:
    input_data[f'Funding Stage_{funding_stage}'] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])
scaled = scaler.transform(input_df)

# Predict and interpret
if st.button("Predict Success Score"):
    score = model.predict(scaled)[0]

    if score < 0 or score > 10:
        st.error("⚠️ Model output is out of expected range. Please check inputs or retrain the model.")
    else:
        st.success(f"🎯 Predicted Success Score: **{score:.2f} / 10**")

        st.progress(min(score / 10, 1.0))  # Progress bar

        # Interpretation
        if score <= 3:
            st.error("❌ This startup has a **high risk** of failure. Consider revisiting the business model or funding.")
        elif score <= 6:
            st.warning("⚠️ This startup has a **moderate risk**. It may succeed with proper execution and market timing.")
        elif score <= 8:
            st.info("✅ This startup shows **good growth potential**. Strategic scaling and funding could lead to success.")
        else:
            st.success("🚀 This startup is **very likely to succeed**! All indicators are strong.")
