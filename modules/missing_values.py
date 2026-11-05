"""
MODULE 3: Missing Value Analysis & Handling
Identifies nulls, calculates missing percentages, visualizes missingness patterns,
and provides scientifically grounded imputation strategies (mean, median, mode, column drop, row drop).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_missing_values(working_df: pd.DataFrame):
    """
    Renders Missing Value Analysis and Cleaning module.
    Returns: updated_df (if changes applied) or None.
    """
    st.markdown('<div class="section-header-box"><h2>Module 3: Missing Value Analysis & Handling</h2><p>Audit incomplete data, inspect missingness distribution, and apply statistically sound imputation methods.</p></div>', unsafe_allow_html=True)

    if working_df.empty:
        st.warning("The active dataset is empty.")
        return None

    # Compute missing values summary
    total_rows = len(working_df)
    missing_counts = working_df.isna().sum()
    missing_pcts = (missing_counts / total_rows) * 100

    missing_summary = pd.DataFrame({
        "Feature Name": working_df.columns,
        "Data Type": working_df.dtypes.astype(str).values,
        "Total Records": total_rows,
        "Missing Values Count": missing_counts.values,
        "Missing Percentage (%)": missing_pcts.round(2).values,
        "Present Records": (total_rows - missing_counts).values
    })

    cols_with_missing = missing_summary[missing_summary["Missing Values Count"] > 0].copy()
    cols_without_missing = missing_summary[missing_summary["Missing Values Count"] == 0].copy()

    total_null_cells = int(missing_counts.sum())

    # Academic Justification Box
    with st.expander("Academic Guide: Missing Data Mechanisms & Treatment Strategies", expanded=True):
        st.markdown("""
        <div class="concept-box">
            <h4>Principles of Missing Value Treatment in Machine Learning</h4>
            <p>Missing data generally falls into three statistical mechanisms:</p>
            <ul>
                <li><b>MCAR (Missing Completely at Random):</b> Missingness is totally independent of observed and unobserved data. Simple imputation or row dropping is unbiased.</li>
                <li><b>MAR (Missing at Random):</b> Missingness depends systematically on observed data (e.g., Age missing more frequently in specific passenger classes).</li>
                <li><b>MNAR (Missing Not at Random):</b> Missingness relates to the unobserved value itself (e.g., higher wealth passengers opting not to record cabin).</li>
            </ul>
            <p><b>Imputation Rules of Thumb:</b></p>
            <ul>
                <li><b>Mean Imputation:</b> Appropriate for normally distributed (symmetric) continuous features without prominent outliers.</li>
                <li><b>Median Imputation:</b> Preferred for skewed numerical distributions (e.g., Titanic <code>Age</code> or <code>Fare</code>) because median is robust to outliers.</li>
                <li><b>Mode Imputation:</b> Mandatory for categorical variables (e.g., Titanic <code>Embarked</code>), replacing missing cells with the most frequent class.</li>
                <li><b>Column Dropping:</b> Justified when missingness exceeds 50%–70% (e.g., Titanic <code>Cabin</code> ~77% missing), where synthetic imputation introduces severe variance and noise.</li>
                <li><b>Avoid Blanket Row Deletion:</b> Never drop all rows with missing values blindly—doing so can decimate sample size and introduce survivorship bias.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Status Overview
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        if total_null_cells == 0:
            st.success(f"**Clean Dataset:** No missing values detected across all {working_df.shape[1]} columns!")
        else:
            st.warning(f"**Missingness Detected:** {total_null_cells:,} missing cells found across {len(cols_with_missing)} columns.")
    with col_stat2:
        st.markdown(f"""
        - **Columns with Missing Values:** `{len(cols_with_missing)}`
        - **Completely Clean Columns:** `{len(cols_without_missing)}`
        """)

    # Visualizations Section
    st.subheader("Missingness Visualisations")
    tab_bar, tab_matrix = st.tabs(["Missing Values Bar Chart", "Missingness Matrix Heatmap"])

    with tab_bar:
        if total_null_cells > 0:
            chart_df = cols_with_missing.sort_values(by="Missing Percentage (%)", ascending=False)
            fig_bar = px.bar(
                chart_df,
                x="Feature Name",
                y="Missing Percentage (%)",
                text="Missing Percentage (%)",
                color="Missing Percentage (%)",
                color_continuous_scale=[[0, '#829B83'], [0.5, '#C08A3E'], [1.0, '#A63A2D']],
                title="Missing Value Percentage per Column (%)"
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_bar.update_layout(
                yaxis_title="Missing (%)",
                xaxis_title="Column",
                yaxis_range=[0, max(100, chart_df["Missing Percentage (%)"].max() + 10)],
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No missing values to display in the bar chart.")

    with tab_matrix:
        if total_null_cells > 0:
            # Subsample up to 250 rows for clear matrix rendering
            sample_n = min(250, len(working_df))
            sample_sub = working_df.iloc[:sample_n].isna().astype(int)
            fig_mat = px.imshow(
                sample_sub,
                labels=dict(x="Feature Name", y="Row Index", color="Is Missing?"),
                color_continuous_scale=[[0, "#E6E7E0"], [1, "#A63A2D"]],
                title=f"Missingness Matrix (First {sample_n} Rows: Ivory=Present, Accent=Missing)",
                aspect="auto"
            )
            fig_mat.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.info("Matrix is 100% complete (no missing cells).")

    # Detailed Summary Table
    st.subheader("Missing Value Audit Breakdown")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"**Incomplete Columns ({len(cols_with_missing)})**")
        if not cols_with_missing.empty:
            st.dataframe(cols_with_missing[["Feature Name", "Data Type", "Missing Values Count", "Missing Percentage (%)"]], use_container_width=True)
        else:
            st.write("None (All columns are 100% complete)")

    with col_t2:
        st.markdown(f"**Clean Columns ({len(cols_without_missing)})**")
        if not cols_without_missing.empty:
            st.dataframe(cols_without_missing[["Feature Name", "Data Type", "Total Records"]], use_container_width=True)
        else:
            st.write("None (All columns contain at least one null)")

    # ----------------- INTERACTIVE CLEANING TOOL -----------------
    st.markdown("---")
    st.subheader("Interactive Missing Value Treatment Tool")

    if total_null_cells == 0:
        st.success("All features in the current working dataset are complete. No missing value imputation is necessary.")
        return None

    col_cfg, col_action = st.columns([1.1, 0.9])

    with col_cfg:
        incomplete_col_names = cols_with_missing["Feature Name"].tolist()
        target_col = st.selectbox("Select Target Column to Clean:", incomplete_col_names)

        is_numeric = pd.api.types.is_numeric_dtype(working_df[target_col])
        curr_missing = int(working_df[target_col].isna().sum())
        curr_pct = (curr_missing / len(working_df)) * 100

        st.caption(f"Target Feature: **`{target_col}`** | Type: **{working_df[target_col].dtype}** | Missing: **{curr_missing} ({curr_pct:.2f}%)**")

        # Dynamic strategy options based on data type
        strategy_options = []
        if is_numeric:
            strategy_options.extend([
                "Median Imputation (Recommended for skewed numerical data)",
                "Mean Imputation (Appropriate for symmetric normal data)",
                "Constant Value Imputation (e.g. 0)"
            ])
        else:
            strategy_options.extend([
                "Mode Imputation (Most frequent category)",
                "Fill with Constant String (e.g. 'Unknown' or 'Missing')"
            ])

        strategy_options.extend([
            "Drop Selected Column (Recommended if missing > 50%)",
            "Drop Rows with Missing Values in this Column",
            "Retain Original Values (Do nothing)"
        ])

        selected_strategy = st.selectbox("Choose Imputation / Cleaning Strategy:", strategy_options)

        custom_val = None
        if "Constant Value Imputation" in selected_strategy:
            custom_val = st.number_input("Enter replacement numerical value:", value=0.0)
        elif "Fill with Constant String" in selected_strategy:
            custom_val = st.text_input("Enter replacement text category:", value="Unknown")

    with col_action:
        st.markdown("#### Scientific Justification & Impact")
        
        # Display contextual justification
        justification_text = ""
        if "Median Imputation" in selected_strategy:
            med_val = working_df[target_col].median()
            justification_text = (
                f"**Why Median?** The median of `{target_col}` is **{med_val:.2f}**. "
                f"Unlike the mean, the median is non-parametric and resistant to extreme outliers or skewness. "
                f"It maintains the central trend without pulling predictions toward extreme values."
            )
        elif "Mean Imputation" in selected_strategy:
            mean_val = working_df[target_col].mean()
            justification_text = (
                f"**Why Mean?** The arithmetic mean of `{target_col}` is **{mean_val:.2f}**. "
                f"It preserves the sample sum and distribution center, but assumes the feature is approximately normally distributed."
            )
        elif "Mode Imputation" in selected_strategy:
            mode_series = working_df[target_col].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A"
            justification_text = (
                f"**Why Mode?** For categorical variable `{target_col}`, the most frequent category is **'{mode_val}'**. "
                f"Because arithmetic operations are undefined on text/nominal categories, the mode is the statistically valid measure of central tendency."
            )
        elif "Drop Selected Column" in selected_strategy:
            justification_text = (
                f"**Why Drop Column?** `{target_col}` has **{curr_pct:.2f}%** missing values. "
                f"When more than 50%–70% of entries are absent, imputing synthetic values creates substantial false information and noise for downstream ML algorithms."
            )
        elif "Drop Rows" in selected_strategy:
            justification_text = (
                f"**Row Dropping:** Will eliminate **{curr_missing}** rows containing NaN in `{target_col}`. "
                f"This preserves genuine data integrity without synthetic imputation, reducing total dataset size to **{len(working_df) - curr_missing}** rows."
            )
        else:
            justification_text = "Retains original missing values as NaN without applying transformations."

        st.info(justification_text)

        # Apply Button
        if st.button("Apply Missing Value Treatment", type="primary", use_container_width=True):
            new_df = working_df.copy()
            audit_message = ""

            if "Median Imputation" in selected_strategy:
                fill_val = new_df[target_col].median()
                new_df[target_col] = new_df[target_col].fillna(fill_val)
                audit_message = f"Imputed missing values in '{target_col}' using Median ({fill_val:.3f})."

            elif "Mean Imputation" in selected_strategy:
                fill_val = new_df[target_col].mean()
                new_df[target_col] = new_df[target_col].fillna(fill_val)
                audit_message = f"Imputed missing values in '{target_col}' using Mean ({fill_val:.3f})."

            elif "Constant Value Imputation" in selected_strategy:
                new_df[target_col] = new_df[target_col].fillna(custom_val)
                audit_message = f"Imputed missing values in '{target_col}' using Constant value ({custom_val})."

            elif "Mode Imputation" in selected_strategy:
                fill_val = new_df[target_col].mode().iloc[0]
                new_df[target_col] = new_df[target_col].fillna(fill_val)
                audit_message = f"Imputed missing values in '{target_col}' using Mode ('{fill_val}')."

            elif "Fill with Constant String" in selected_strategy:
                new_df[target_col] = new_df[target_col].fillna(custom_val)
                audit_message = f"Imputed missing values in '{target_col}' with String '{custom_val}'."

            elif "Drop Selected Column" in selected_strategy:
                new_df = new_df.drop(columns=[target_col])
                audit_message = f"Dropped column '{target_col}' due to high missingness ({curr_pct:.2f}%)."

            elif "Drop Rows" in selected_strategy:
                rows_before = len(new_df)
                new_df = new_df.dropna(subset=[target_col])
                rows_after = len(new_df)
                audit_message = f"Dropped {rows_before - rows_after} rows with missing values in '{target_col}'."

            else:
                audit_message = f"Retained original values for '{target_col}'."

            st.success(f"**Successfully Applied:** {audit_message}")
            return new_df, audit_message

    return None

