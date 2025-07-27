import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import joblib
import os

# ---------------- File Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLS_PATH = os.path.join(BASE_DIR, "input_columns.pkl")
CSV_MAIN = os.path.join(BASE_DIR, "startup_predictions-offline.csv")
CSV_GEO = os.path.join(BASE_DIR, "Final-startup_success_predictions.csv")
CSV_FEAT = os.path.join(BASE_DIR, "feature_importance.csv")

# ---------------- Model Load ----------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(COLS_PATH, "rb") as f:
        input_columns = joblib.load(f)
except FileNotFoundError as e:
    st.error(f"🚨 Required model/scaler file not found: {e}")
    st.stop()

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Startup Dashboard & Predictor", layout="wide")

with st.sidebar:
    page = option_menu("Startup Dashboard",
                       ["Overview", "Profile & Geography", "Model Insights", "Predict Success"],
                       icons=["house", "globe", "bar-chart-line", "rocket"],
                       menu_icon="cast", default_index=0)

# ---------------- Data Load ----------------
@st.cache_data
def load_data():
    try:
        df_main = pd.read_csv(CSV_MAIN)
        df_geo = pd.read_csv(CSV_GEO)
        df_imp = pd.read_csv(CSV_FEAT)
        return df_main, df_geo, df_imp
    except FileNotFoundError as e:
        st.error(f"🚨 Required CSV file not found: {e}")
        st.stop()

# ---------------- Reverse One-Hot Encoding ----------------
def reverse_one_hot(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix + "_")]
    if cols:
        df[prefix] = df[cols].idxmax(axis=1).str.replace(f"{prefix}_", "")
    return df

# ---------------- Main Logic ----------------
if page != "Predict Success":
    df, df_geo, df_imp = load_data()

    for col in ["Country", "Industry", "Funding Stage"]:
        df = reverse_one_hot(df, col)
        df_geo = reverse_one_hot(df_geo, col)

    if "Startup Age" not in df_geo.columns and "Founded Year" in df_geo.columns:
        df_geo["Startup Age"] = 2025 - df_geo["Founded Year"]

    if "Predicted Category" not in df_geo.columns and "Predicted" in df_geo.columns:
        df_geo["Predicted Category"] = df_geo["Predicted"].map({0: "Low", 1: "Medium", 2: "High"})

# -------------- Overview Page --------------
if page == "Overview":
    st.title("🚀 Startup Overview + Success Analysis")
    st.subheader("Filter Data")
    df_filtered = df.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", ["All"] + sorted(df["Country"].dropna().unique().tolist()))
        if country != "All":
            df_filtered = df_filtered[df_filtered["Country"] == country]

    with c2:
        industry = st.selectbox("Industry", ["All"] + sorted(df["Industry"].dropna().unique().tolist()))
        if industry != "All":
            df_filtered = df_filtered[df_filtered["Industry"] == industry]

    with c3:
        stage = st.selectbox("Funding Stage", ["All"] + sorted(df["Funding Stage"].dropna().unique().tolist()))
        if stage != "All":
            df_filtered = df_filtered[df_filtered["Funding Stage"] == stage]

    if df_filtered.empty:
        st.warning("No records match the selected filters.")
    else:
        st.subheader("📌 Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Startups", len(df_filtered))
        m2.metric("Avg Funding ($M)", f"{df_filtered['Total Funding ($M)'].mean():,.2f}")
        m3.metric("Avg Valuation ($B)", f"{df_filtered['Valuation ($B)'].mean():,.2f}")
        m4.metric("Total Acquisitions", int(df_filtered["Acquired?"].sum()))

        st.subheader("📊 Success Category vs Funding Stage")
        col1, col2 = st.columns(2)
        with col1:
            if "Predicted Category" in df_filtered.columns:
                fig_pie = px.pie(df_filtered, names="Predicted Category", title="Success Category Distribution", hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            if "Funding Stage" in df_filtered.columns and "Predicted Category" in df_filtered.columns:
                fig_bar = px.histogram(df_filtered, x="Funding Stage", color="Predicted Category", barmode="group",
                                       category_orders={"Predicted Category": ["Low", "Medium", "High"]})
                st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📌 Success by Industry (Treemap)")
        if "Industry" in df_filtered.columns and "Total Funding ($M)" in df_filtered.columns:
            fig_tree = px.treemap(df_filtered,
                                  path=[px.Constant("All"), "Industry", "Predicted Category"],
                                  values="Total Funding ($M)", title="Treemap of Industry Success")
            st.plotly_chart(fig_tree, use_container_width=True)

# -------------- Predict Success Page --------------
elif page == "Predict Success":
    st.title("🚀 Startup Growth Success Score Predictor")
    st.markdown("Enter the details of your startup to predict its success potential.")

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

    # Dropdowns for one-hot features
    ohe_country = st.selectbox("Country", sorted({col.split("_")[1] for col in input_columns if col.startswith("Country_")}))
    ohe_industry = st.selectbox("Industry", sorted({col.split("_")[1] for col in input_columns if col.startswith("Industry_")}))
    ohe_stage = st.selectbox("Funding Stage", sorted({col.split("_")[1] for col in input_columns if col.startswith("Funding Stage_")}))

    if st.button("🔍 Predict Success"):
        try:
            startup_age = 2025 - founded_year

            input_dict = {col: 0 for col in input_columns}

            input_dict["Total Funding ($M)"] = funding
            input_dict["Number of Employees"] = employees
            input_dict["Annual Revenue ($M)"] = revenue
            input_dict["Valuation ($B)"] = valuation
            input_dict["Social Media Followers"] = followers
            input_dict["Tech Stack Count"] = tech_stack
            input_dict["Acquired?"] = acquired
            input_dict["IPO?"] = ipo
            input_dict["Startup Age"] = startup_age
            input_dict["Customer Base (Millions)"] = customer_base

            if f"Country_{ohe_country}" in input_dict:
                input_dict[f"Country_{ohe_country}"] = 1
            if f"Industry_{ohe_industry}" in input_dict:
                input_dict[f"Industry_{ohe_industry}"] = 1
            if f"Funding Stage_{ohe_stage}" in input_dict:
                input_dict[f"Funding Stage_{ohe_stage}"] = 1

            final_input = pd.DataFrame([input_dict])

            scaled_input = scaler.transform(final_input)
            prediction = model.predict(scaled_input)[0]
            probas = model.predict_proba(scaled_input)[0]

            pred_label = {0: "Low", 1: "Medium", 2: "High"}.get(prediction, "Unknown")
            st.success(f"🧠 **Predicted Success Category: {pred_label}**")

            st.subheader("📊 Prediction Probabilities")
            st.write({
                "Low": f"{probas[0]*100:.2f}%",
                "Medium": f"{probas[1]*100:.2f}%",
                "High": f"{probas[2]*100:.2f}%"
            })

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
