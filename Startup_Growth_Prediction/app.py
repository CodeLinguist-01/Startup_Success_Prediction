import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(page_title="Startup Dashboard", layout="wide")

# Sidebar navigation
with st.sidebar:
    page = option_menu("Startup Dashboard",
                       ["Overview", "Profile & Geography", "Model Insights"],
                       icons=["house", "globe", "bar-chart-line"],
                       menu_icon="cast", default_index=0)

# Load CSV files
@st.cache_data
def load_data():
    try:
        df_main = pd.read_csv("startup_predictions-offline.csv")
        df_geo = pd.read_csv("Final-startup_success_predictions.csv")
        df_imp = pd.read_csv("feature_importance.csv")
        
        return df_main, df_geo, df_imp
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

df, df_geo, df_imp = load_data()

# Convert one-hot encoded columns back to categorical
def reverse_one_hot(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix + "_")]
    if cols:
        df[prefix] = df[cols].idxmax(axis=1).str.replace(f"{prefix}_", "")
    return df

for col in ["Country", "Industry", "Funding Stage"]:
    df = reverse_one_hot(df, col)
    df_geo = reverse_one_hot(df_geo, col)

# Handle Startup Age and Predicted Category
if "Startup Age" not in df_geo.columns and "Founded Year" in df_geo.columns:
    df_geo["Startup Age"] = 2025 - df_geo["Founded Year"]

if "Predicted Category" not in df_geo.columns and "Predicted" in df_geo.columns:
    df_geo["Predicted Category"] = df_geo["Predicted"].map({0: "Low", 1: "Medium", 2: "High"})

# ---------------- Overview Page ----------------
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

# ---------------- Profile & Geography Page ----------------

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

# ---------------- Model Insights Page ----------------
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
