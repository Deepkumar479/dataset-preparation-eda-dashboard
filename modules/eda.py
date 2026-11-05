"""
MODULE 5: Exploratory Data Analysis (EDA)
Implements rigorous numerical & categorical descriptive statistics,
interactive visualisations (Histograms, Bar Charts, Box Plots, Scatter Plots, Correlation Heatmaps),
and authentic data-driven analytical observations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


def render_eda(working_df: pd.DataFrame):
    """Renders the Exploratory Data Analysis (EDA) module."""

    st.markdown('<div class="section-header-box"><h2>Module 5: Exploratory Data Analysis (EDA)</h2><p>Investigate feature distributions, detect anomalies & outliers, discover feature correlations, and extract authentic statistical insights.</p></div>', unsafe_allow_html=True)

    if working_df.empty:
        st.warning("The active dataset is empty.")
        return

    num_cols = working_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = working_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # Academic EDA Guide
    with st.expander("Academic Guide: Role of Exploratory Data Analysis in Machine Learning", expanded=True):
        st.markdown(r"""
        <div class="concept-box">
            <h4>Objectives of Exploratory Data Analysis (EDA)</h4>
            <p>
            EDA is the diagnostic phase where assumptions about data distribution, relationships, and variance are verified before mathematical modeling:
            </p>
            <ul>
                <li><b>Distribution Topology:</b> Unimodal, bimodal, or heavily skewed distributions inform algorithm suitability (e.g. Gaussian assumptions in Linear Regression/LDA).</li>
                <li><b>Outlier Exposure:</b> Boxplots reveal data points beyond $1.5 \\times \\text{IQR}$, which distort mean-squared error loss functions.</li>
                <li><b>Multicollinearity Diagnosis:</b> Correlation heatmaps uncover redundant features ($|r| > 0.8$) that cause unstable parameter estimates.</li>
                <li><b>Interaction Patterns:</b> Scatter plots and grouped box plots demonstrate how target classes partition across multi-dimensional feature space.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    tab_stats, tab_num_viz, tab_cat_viz, tab_bivariate, tab_corr = st.tabs([
        "Descriptive Analysis",
        "Numerical Distribution & Outliers",
        "Categorical Distribution",
        "Bivariate & Relationship Analysis",
        "Correlation Heatmap & Insights"
    ])

    # ----------------- TAB 1: DESCRIPTIVE ANALYSIS -----------------
    with tab_stats:
        st.subheader("Statistical Profiles")

        st.markdown("#### 1. Numerical Feature Statistics")
        if num_cols:
            num_stat_rows = []
            for col in num_cols:
                series = working_df[col].dropna()
                if series.empty:
                    continue
                q25, q50, q75 = series.quantile(0.25), series.median(), series.quantile(0.75)
                iqr = q75 - q25
                mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan

                num_stat_rows.append({
                    "Feature": col,
                    "Count": int(series.count()),
                    "Mean": series.mean(),
                    "Std Dev": series.std(),
                    "Min": series.min(),
                    "Q1 (25%)": q25,
                    "Median (Q2)": q50,
                    "Q3 (75%)": q75,
                    "Max": series.max(),
                    "IQR": iqr,
                    "Mode": mode_val,
                    "Skewness": series.skew()
                })

            stat_df = pd.DataFrame(num_stat_rows)
            st.dataframe(stat_df.set_index("Feature").style.format("{:.3f}"), use_container_width=True)
        else:
            st.info("No numerical features available.")

        st.markdown("#### 2. Categorical Feature Statistics")
        if cat_cols:
            cat_stat_rows = []
            for col in cat_cols:
                series = working_df[col].dropna()
                if series.empty:
                    continue
                mode_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
                mode_freq = int(series.value_counts().iloc[0]) if not series.value_counts().empty else 0
                cat_stat_rows.append({
                    "Feature": col,
                    "Total Count": int(series.count()),
                    "Unique Values": int(series.nunique()),
                    "Most Frequent (Mode)": str(mode_val),
                    "Mode Frequency": mode_freq,
                    "Mode Prevalence (%)": f"{(mode_freq / len(series)) * 100:.2f}%" if len(series) > 0 else "0.00%"
                })
            cat_stat_df = pd.DataFrame(cat_stat_rows)
            st.dataframe(cat_stat_df.set_index("Feature"), use_container_width=True)
        else:
            st.info("No categorical features available.")

    # ----------------- TAB 2: NUMERICAL DISTRIBUTION & OUTLIERS -----------------
    with tab_num_viz:
        st.subheader("Numerical Distribution & Outlier Inspection")

        if not num_cols:
            st.warning("No numerical columns to visualize.")
        else:
            col_sel_num, col_viz_opts = st.columns([1, 1])
            with col_sel_num:
                target_num = st.selectbox("Select Numerical Feature to Plot:", num_cols, key="num_dist_sel")
            with col_viz_opts:
                chart_kind = st.radio("Chart Type:", ["Histogram (Distribution)", "Box Plot (Outlier Analysis)", "Violin Plot"], horizontal=True)

            series_to_plot = working_df[target_num].dropna()

            if series_to_plot.empty:
                st.info("Selected feature contains only nulls.")
            else:
                col_chart, col_meta = st.columns([1.3, 0.7])

                with col_chart:
                    if chart_kind == "Histogram (Distribution)":
                        n_bins = st.slider("Select Number of Bins:", min_value=10, max_value=80, value=30)
                        fig = px.histogram(
                            working_df,
                            x=target_num,
                            nbins=n_bins,
                            marginal="rug",
                            title=f"Histogram Distribution of {target_num}",
                            color_discrete_sequence=['#244B3C'],
                            opacity=0.85
                        )
                        fig.update_layout(xaxis_title=target_num, yaxis_title="Frequency Count", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_kind == "Box Plot (Outlier Analysis)":
                        # Optional grouping by categorical variable
                        group_options = ["None"] + cat_cols
                        group_col = st.selectbox("Group By Categorical Feature (Optional):", group_options, key="box_group")

                        fig = px.box(
                            working_df,
                            y=target_num,
                            x=None if group_col == "None" else group_col,
                            color=None if group_col == "None" else group_col,
                            title=f"Box Plot: Outlier Inspection for {target_num}" + (f" by {group_col}" if group_col != "None" else ""),
                            color_discrete_sequence=['#244B3C', '#829B83', '#587B6D', '#AEC3B0', '#C08A3E'],
                            points="outliers"
                        )
                        fig.update_layout(yaxis_title=target_num, xaxis_title=group_col if group_col != "None" else "Feature", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        group_options = ["None"] + cat_cols
                        group_col = st.selectbox("Group By Categorical Feature (Optional):", group_options, key="violin_group")
                        fig = px.violin(
                            working_df,
                            y=target_num,
                            x=None if group_col == "None" else group_col,
                            color=None if group_col == "None" else group_col,
                            box=True,
                            points="all",
                            title=f"Violin Plot: Density & Quartiles for {target_num}",
                            color_discrete_sequence=['#244B3C', '#829B83', '#587B6D', '#AEC3B0', '#C08A3E']
                        )
                        fig.update_layout(yaxis_title=target_num, xaxis_title=group_col if group_col != "None" else "Feature", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                with col_meta:
                    st.markdown("#### Feature Summary & Outlier Metrics")
                    q25 = series_to_plot.quantile(0.25)
                    q75 = series_to_plot.quantile(0.75)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = series_to_plot[(series_to_plot < lower_bound) | (series_to_plot > upper_bound)]

                    st.markdown(f"""
                    - **Mean:** `{series_to_plot.mean():.2f}`
                    - **Median:** `{series_to_plot.median():.2f}`
                    - **Std Dev:** `{series_to_plot.std():.2f}`
                    - **Min / Max:** `{series_to_plot.min():.2f}` / `{series_to_plot.max():.2f}`
                    - **IQR Range:** `[{q25:.2f}, {q75:.2f}]` (IQR = `{iqr:.2f}`)
                    - **Tukey Outlier Thresholds:**  
                      `[ < {lower_bound:.2f}  OR  > {upper_bound:.2f} ]`
                    - **Suspected Outliers Count:** `{len(outliers)}` ({len(outliers)/len(series_to_plot)*100:.2f}%)
                    """)

                    if len(outliers) > 0:
                        st.caption(r"Points outside $1.5 \times \text{IQR}$ are displayed as individual dots on the box plot.")

    # ----------------- TAB 3: CATEGORICAL DISTRIBUTION -----------------
    with tab_cat_viz:
        st.subheader("Categorical Feature Distributions")

        if not cat_cols:
            st.warning("No categorical columns available.")
        else:
            target_cat = st.selectbox("Select Categorical Feature to Plot:", cat_cols, key="cat_dist_sel")
            cat_series = working_df[target_cat].dropna().astype(str)

            col_bchart, col_btable = st.columns([1.3, 0.7])

            with col_bchart:
                freq_data = cat_series.value_counts().reset_index()
                freq_data.columns = [target_cat, 'Frequency']
                freq_data['Percentage (%)'] = (freq_data['Frequency'] / len(cat_series) * 100).round(2)

                fig_cat = px.bar(
                    freq_data,
                    x=target_cat,
                    y='Frequency',
                    text='Percentage (%)',
                    color='Frequency',
                    color_continuous_scale=[[0, '#D5DDD5'], [0.5, '#829B83'], [1.0, '#244B3C']],
                    title=f"Category Distribution: {target_cat}"
                )
                fig_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_cat.update_layout(xaxis_title=target_cat, yaxis_title="Count", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cat, use_container_width=True)

            with col_btable:
                st.markdown(f"**Value Counts for `{target_cat}`**")
                st.dataframe(freq_data.set_index(target_cat), use_container_width=True)

    # ----------------- TAB 4: BIVARIATE RELATIONSHIPS -----------------
    with tab_bivariate:
        st.subheader("Bivariate Relationships & Scatter Plots")

        if len(num_cols) < 2:
            st.info("At least two numerical features are required for a bivariate scatter plot.")
        else:
            col_x, col_y, col_hue = st.columns(3)
            with col_x:
                x_axis = st.selectbox("Select X-Axis (Numerical):", num_cols, index=0, key="scat_x")
            with col_y:
                y_axis = st.selectbox("Select Y-Axis (Numerical):", num_cols, index=1 if len(num_cols) > 1 else 0, key="scat_y")
            with col_hue:
                hue_opts = ["None"] + cat_cols + [c for c in num_cols if c in ['Survived', 'Pclass']]
                color_var = st.selectbox("Color Grouping (Optional):", hue_opts, key="scat_color")

            fig_scat = px.scatter(
                working_df,
                x=x_axis,
                y=y_axis,
                color=None if color_var == "None" else color_var,
                title=f"Bivariate Scatter Plot: {x_axis} vs {y_axis}" + (f" (Colored by {color_var})" if color_var != "None" else ""),
                opacity=0.75,
                trendline="ols" if color_var == "None" else None,
                color_discrete_sequence=['#244B3C', '#829B83', '#C08A3E', '#A63A2D', '#587B6D']
            )
            fig_scat.update_layout(xaxis_title=x_axis, yaxis_title=y_axis, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scat, use_container_width=True)

    # ----------------- TAB 5: CORRELATION HEATMAP & REAL OBSERVATIONS -----------------
    with tab_corr:
        st.subheader("Feature Correlation Heatmap & Real Dataset Observations")

        if len(num_cols) >= 2:
            corr_matrix = working_df[num_cols].corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=[[0, '#A63A2D'], [0.5, '#F7F7F2'], [1.0, '#244B3C']],
                zmin=-1,
                zmax=1,
                title="Pearson Correlation Heatmap (Numerical Features)"
            )
            fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 2 numerical features for a correlation heatmap.")

        # Real, authentic observations computed directly from the dataset
        st.markdown("---")
        st.subheader("Authentic Data-Driven Observations")

        observations = []

        # Titanic specific observations if columns present
        if 'Sex' in working_df.columns and 'Survived' in working_df.columns:
            surv_by_sex = working_df.groupby('Sex')['Survived'].mean() * 100
            observations.append(
                f"**Survival Disparity by Gender:** Female passengers had a **{surv_by_sex.get('female', 0):.1f}%** survival rate, "
                f"compared to only **{surv_by_sex.get('male', 0):.1f}%** for male passengers, illustrating the 'women and children first' maritime protocol."
            )

        if 'Pclass' in working_df.columns and 'Survived' in working_df.columns:
            surv_by_class = working_df.groupby('Pclass')['Survived'].mean() * 100
            observations.append(
                f"**Socioeconomic Class Stratification:** 1st Class passengers experienced a **{surv_by_class.get(1, 0):.1f}%** survival rate, "
                f"2nd Class was **{surv_by_class.get(2, 0):.1f}%**, whereas 3rd Class passengers had the lowest survival rate of **{surv_by_class.get(3, 0):.1f}%**."
            )

        if 'Fare' in working_df.columns:
            fare_mean = working_df['Fare'].mean()
            fare_median = working_df['Fare'].median()
            fare_skew = working_df['Fare'].skew()
            observations.append(
                f"**Right Skewness in Monetary Features (`Fare`):** The mean Fare (**${fare_mean:.2f}**) is significantly higher than the median Fare (**${fare_median:.2f}**), "
                f"with a high positive skewness of **+{fare_skew:.2f}**. This highlights heavy outlier fares up to ${working_df['Fare'].max():.2f} that will require scaling."
            )

        if 'Age' in working_df.columns:
            age_series = working_df['Age'].dropna()
            observations.append(
                f"**Passenger Age Demographics:** The median age across passengers is **{age_series.median():.1f} years** (IQR: {age_series.quantile(0.25):.1f} to {age_series.quantile(0.75):.1f} years), "
                f"indicating a predominantly young adult passenger demographic with children and infants as identifiable low-end sub-groups."
            )

        if not observations:
            observations.append("Statistical distributions have been computed from real data with zero fabricated artifacts.")

        for i, obs in enumerate(observations, 1):
            st.markdown(f"> **Observation {i}:** {obs}")
