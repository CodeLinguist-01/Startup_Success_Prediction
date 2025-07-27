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

# -------------- Profile & Geography Page --------------
elif page == "Profile & Geography":
    st.title("🌍 Startup Profile & Geography")
    df_filtered_geo = df_geo.copy()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        country = st.selectbox("Country", ["All"] + sorted(df_geo["Country"].dropna().unique().tolist()))
        if country != "All":
            df_filtered_geo = df_filtered_geo[df_filtered_geo["Country"] == country]

    with col2:
        industry = st.selectbox("Industry", ["All"] + sorted(df_geo["Industry"].dropna().unique().tolist()))
        if industry != "All":
            df_filtered_geo = df_filtered_geo[df_filtered_geo["Industry"] == industry]

    with col3:
        stage = st.selectbox("Funding Stage", ["All"] + sorted(df_geo["Funding Stage"].dropna().unique().tolist()))
        if stage != "All":
            df_filtered_geo = df_filtered_geo[df_filtered_geo["Funding Stage"] == stage]

    with col4:
        pred_cat = st.selectbox("Predicted Category", ["All", "Low", "Medium", "High"])
        if pred_cat != "All":
            df_filtered_geo = df_filtered_geo[df_filtered_geo["Predicted Category"] == pred_cat]

    if df_filtered_geo.empty:
        st.warning("No data for selected filters.")
    else:
        st.subheader("🗺️ Global Startup Spread")
        if "Country" in df_filtered_geo.columns:
            fig_map = px.scatter_geo(df_filtered_geo, locations="Country", locationmode='country names',
                                     size="Total Funding ($M)", color="Country",
                                     hover_name="Country", title="Startup Spread by Country")
            st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("🏭 Avg Funding by Industry")
        if "Industry" in df_filtered_geo.columns:
            avg_funding = df_filtered_geo.groupby("Industry")["Total Funding ($M)"].mean().reset_index()
            fig_ind = px.bar(avg_funding, x="Industry", y="Total Funding ($M)", title="Avg Funding by Industry")
            st.plotly_chart(fig_ind, use_container_width=True)

        st.subheader("📈 Age vs Valuation")
        if "Startup Age" in df_filtered_geo.columns and "Valuation ($B)" in df_filtered_geo.columns:
            fig_scatter = px.scatter(df_filtered_geo, x="Startup Age", y="Valuation ($B)",
                                     color="Predicted Category", hover_data=["Country", "Industry"],
                                     title="Startup Age vs Valuation")
            st.plotly_chart(fig_scatter, use_container_width=True)

# -------------- Model Insights Page --------------
elif page == "Model Insights":
    st.title("📊 Model Insights & Feature Importance")

    st.subheader("🧠 Feature Importance")
    fig_feat = px.bar(df_imp.sort_values(by="Importance", ascending=True),
                      x="Importance", y="Feature", orientation="h",
                      title="Feature Importance (Low to High)")
    st.plotly_chart(fig_feat, use_container_width=True)

    st.subheader("📋 Complete Dataset (Offline Predictions)")
    with st.expander("View Full Dataset"):
        st.dataframe(df)

# -------------- Predict Success Page --------------
elif page == "Predict Success":
    st.title("🚀 Startup Growth Success Score Predictor")
    st.markdown("Enter the details of your startup to predict its success potential.")

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

    if st.button("Predict Success Score"):
        try:
            age = 2025 - founded_year
            input_dict = {
                "Total Funding ($M)": funding,
                "Employees": employees,
                "Annual Revenue ($M)": revenue,
                "Valuation ($B)": valuation,
                "Social Media Followers": followers,
                "Tech Stack Count": tech_stack,
                "Acquired?": acquired,
                "IPO?": ipo,
                "Customer Base (Millions)": customer_base,
                "Founded Year": founded_year
            }

            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=input_columns, fill_value=0)
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]

            category_map = {0: "Low", 1: "Medium", 2: "High"}
            st.success(f"🎯 **Predicted Startup Success Category:** {category_map.get(pred, 'Unknown')}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
