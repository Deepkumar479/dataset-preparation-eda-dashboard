"""
MODULE 1: Dashboard Overview
Provides project metadata, dataset selector, core dataset KPIs,
and an overview of the Unit I Machine Learning preparation lifecycle.
"""

import streamlit as st
import pandas as pd


def render_overview(raw_df: pd.DataFrame, working_df: pd.DataFrame, dataset_name: str, source_info: str):
    """Renders the Dashboard Overview module."""

    st.markdown("""
    <div class="academic-header">
        <h1>Dataset Preparation and Exploratory Data Analysis (EDA) Dashboard</h1>
        <p class="subtitle">An Interactive System for Data Inspection, Cleaning, Analysis and Preprocessing for Machine Learning</p>
        <div class="badge-container">
            <span class="academic-badge">Academic Project</span>
            <span class="academic-badge">B.Tech AI & Data Science</span>
            <span class="academic-badge">Machine Learning – Unit I</span>
            <span class="academic-badge">CIE 2: Activity 1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Academic Notice
    st.info(
        "**Unit I Scope Notice:** This application strictly focuses on exploratory data analysis, data quality inspection, "
        "missing-value imputation, data consistency, categorical encoding, and feature scaling. "
        "In accordance with Unit I curriculum objectives, no predictive models (e.g., Regression, Decision Trees) are included."
    )

    # Metric Cards Computation
    total_rows = len(working_df)
    total_cols = working_df.shape[1]
    num_cols = len(working_df.select_dtypes(include=['number']).columns)
    cat_cols = len(working_df.select_dtypes(include=['object', 'category']).columns)
    total_missing = int(working_df.isna().sum().sum())
    total_duplicates = int(working_df.duplicated().sum())

    st.markdown('<div class="section-header-box"><h2>Dataset Health & Key Indicators</h2><p>Real-time dataset dimensions and data quality metrics for the active working data.</p></div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Records", f"{total_rows:,}", help="Total number of observations (rows)")
    with col2:
        st.metric("Total Columns", f"{total_cols}", help="Total number of features (columns)")
    with col3:
        st.metric("Numerical", f"{num_cols}", help="Continuous & discrete numerical features")
    with col4:
        st.metric("Categorical", f"{cat_cols}", help="Nominal, ordinal & text features")
    with col5:
        st.metric("Missing Values", f"{total_missing:,}", delta=None if total_missing == 0 else f"{total_missing} nulls", delta_color="inverse", help="Total count of null (NaN) cells")
    with col6:
        st.metric("Duplicates", f"{total_duplicates:,}", delta=None if total_duplicates == 0 else f"{total_duplicates} rows", delta_color="inverse", help="Identical duplicate records")

    st.markdown("---")

    # Dataset Source & Description
    col_meta, col_scheme = st.columns([1.1, 0.9])

    with col_meta:
        st.subheader("Dataset Details")
        st.markdown(f"""
        - **Dataset Name:** `{dataset_name}`
        - **Source:** [{source_info}]({source_info if source_info.startswith('http') else '#'})
        - **Original Dimensions:** `{raw_df.shape[0]}` rows × `{raw_df.shape[1]}` columns
        - **Current Working Dimensions:** `{working_df.shape[0]}` rows × `{working_df.shape[1]}` columns
        - **State Status:** {"Raw Dataset (No transformations applied yet)" if raw_df.equals(working_df) else "Transformed Pipeline Active (Modifications recorded)"}
        """)

        # Quick Preview
        with st.expander("Quick Peek at Current Working Data (First 3 rows)", expanded=False):
            st.dataframe(working_df.head(3), use_container_width=True)

    with col_scheme:
        st.subheader("Unit I Evaluation Scheme (5 Marks)")
        st.markdown("""
        | Criterion | Scope & Tasks | Marks |
        | :--- | :--- | :---: |
        | **1. Dataset Understanding** | Domain context, `head()`, `info()`, `describe()`, data types | 1.0 |
        | **2. Data-Quality Handling** | Missing value detection & imputation, duplicates, consistency | 1.0 |
        | **3. Exploratory Data Analysis** | Histograms, boxplots, scatter plots, correlation heatmaps | 1.0 |
        | **4. Feature Preprocessing** | One-Hot / Ordinal Encoding, StandardScaler / MinMaxScaler | 1.0 |
        | **5. Prepared Dataset & Report**| Prepared dataset verification, pipeline audit log & export | 1.0 |
        """)

    # Pipeline Transformation Notice
    if not raw_df.equals(working_df):
        st.success(
            f"**Pipeline Active:** You have applied data transformations. "
            f"The working dataset now has {working_df.shape[0]} rows and {working_df.shape[1]} columns. "
            f"The original dataset remains safe and unaltered in memory."
        )
    else:
        st.info("The working dataset is currently identical to the original input dataset. Proceed through Modules 2 to 7 to inspect, clean, encode, and scale the data.")

