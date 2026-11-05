"""
MODULE 7: Numerical Feature Scaling
Applies StandardScaler (Standardisation) and MinMaxScaler (Normalisation).
Excludes identifiers/targets, displays before-and-after distribution metrics,
and educates students on preventing data leakage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def render_scaling(working_df: pd.DataFrame):
    """
    Renders Numerical Feature Scaling module.
    Returns: (updated_df, audit_message) if changes applied, else None.
    """
    st.markdown('<div class="section-header-box"><h2>Module 7: Numerical Feature Scaling</h2><p>Normalize or standardize numerical feature magnitudes to eliminate scale bias in distance- and gradient-based algorithms.</p></div>', unsafe_allow_html=True)

    if working_df.empty:
        st.warning("The active dataset is empty.")
        return None

    # Academic Theory Box
    with st.expander("Academic Guide: Why Feature Scaling is Essential in Machine Learning", expanded=True):
        st.markdown(r"""
        <div class="concept-box">
            <h4>Mathematical Foundations of Feature Scaling</h4>
            <p>
            Features measured in vastly different numerical domains (e.g., <code>Age: [0, 80]</code> vs. <code>Fare: [0, 512]</code>) 
            wreak havoc on mathematical optimization:
            </p>
            <ul>
                <li><b>Distance Metric Distortion:</b> Distance-based algorithms (KNN, SVM, K-Means Clustering, PCA) compute Euclidean distance:
                $$d(\\mathbf{p}, \\mathbf{q}) = \\sqrt{\\sum_{i=1}^n (p_i - q_i)^2}$$
                A difference of $50 in <code>Fare</code> will completely overpower a 10-year difference in <code>Age</code> simply due to unit magnitude!</li>
                <li><b>Gradient Descent Convergence:</b> In neural networks and linear models with gradient descent, unscaled features produce highly elliptical loss contours, causing oscillating, slow convergence. Scaled features produce spherical contours enabling direct, rapid descent.</li>
                <li><b>StandardScaler (Standardisation):</b> Centers data around $\\mu = 0$ with $\\sigma = 1$:
                $$z = \\frac{x - \\mu}{\\sigma}$$
                Preserves outlier information and works best for Gaussian-like features.</li>
                <li><b>MinMaxScaler (Normalisation):</b> Bounds values strictly within $[0, 1]$:
                $$x' = \\frac{x - x_{min}}{x_{max} - x_{min}}$$
                Essential when positive bounds are needed (e.g. image pixels, algorithms requiring $[0, 1]$ bounds). Highly sensitive to extreme outliers.</li>
                <li><b>CRITICAL DATA LEAKAGE RULE:</b> In an end-to-end predictive machine learning pipeline, 
                scalers must be <b>fit only on the training set</b> (<code>scaler.fit(X_train)</code>) and subsequently applied to transform test/validation sets (<code>scaler.transform(X_test)</code>). Computing min/max or mean/std across the entire dataset leaks validation distribution parameters into training.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Detect numerical columns
    all_num_cols = working_df.select_dtypes(include=['number']).columns.tolist()

    # Automatically filter out identifiers and binary targets
    excluded_defaults = ['PassengerId', 'passengerid', 'id', 'ID', 'Survived', 'survived', 'target', 'label']
    default_scaling_candidates = [c for c in all_num_cols if c.lower() not in [x.lower() for x in excluded_defaults] and working_df[c].nunique() > 2]

    if not all_num_cols:
        st.warning("No numerical features found in the current dataset to scale.")
        return None

    st.subheader("Select Features and Scaling Algorithm")

    col_cfg1, col_cfg2 = st.columns([1.1, 0.9])

    with col_cfg1:
        selected_scale_cols = st.multiselect(
            "Select Numerical Columns to Scale:",
            options=all_num_cols,
            default=default_scaling_candidates if default_scaling_candidates else all_num_cols[:2]
        )

        scaler_type = st.radio(
            "Select Scaling Method:",
            ["StandardScaler (Standardisation: Mean=0, Std=1)", "MinMaxScaler (Normalisation: Range [0, 1])"],
            horizontal=False
        )

        scaling_mode = st.radio(
            "Output Mode:",
            ["Replace original columns in-place", "Append as new scaled columns (e.g. 'Age_scaled')"],
            horizontal=True
        )

    with col_cfg2:
        st.markdown("#### Current Feature Ranges (Before Scaling)")
        if selected_scale_cols:
            range_info = []
            for c in selected_scale_cols:
                ser = working_df[c].dropna()
                range_info.append({
                    "Feature": c,
                    "Min": ser.min(),
                    "Max": ser.max(),
                    "Mean": ser.mean(),
                    "Std Dev": ser.std()
                })
            st.dataframe(pd.DataFrame(range_info).set_index("Feature").style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("Select at least one numerical feature to display baseline ranges.")

    if not selected_scale_cols:
        return None

    # Check for NaNs before scaling
    has_nans = working_df[selected_scale_cols].isna().sum().sum() > 0
    if has_nans:
        st.error(
            "⚠️ Selected features contain missing values (NaNs). "
            "Scikit-learn scalers cannot process NaN values. Please return to Module 3 (Missing Values) to impute missing cells before scaling."
        )
        return None

    st.markdown("---")
    if st.button("Apply Numerical Feature Scaling", type="primary"):
        new_df = working_df.copy()
        
        if "StandardScaler" in scaler_type:
            scaler = StandardScaler()
            method_name = "StandardScaler"
        else:
            scaler = MinMaxScaler()
            method_name = "MinMaxScaler"

        scaled_values = scaler.fit_transform(new_df[selected_scale_cols])

        created_or_modified_cols = []
        if "Replace original columns in-place" in scaling_mode:
            new_df[selected_scale_cols] = scaled_values
            created_or_modified_cols = selected_scale_cols
            audit_msg = f"Applied {method_name} in-place to features: {selected_scale_cols}."
        else:
            for i, col in enumerate(selected_scale_cols):
                new_col_name = f"{col}_scaled"
                new_df[new_col_name] = scaled_values[:, i]
                created_or_modified_cols.append(new_col_name)
            audit_msg = f"Applied {method_name} creating new features: {created_or_modified_cols}."

        st.success(f"**Scaling Successful!** {audit_msg}")

        # Before & After Statistics Comparison
        st.subheader("Before vs. After Statistics Comparison")
        comp_rows = []
        for i, orig_col in enumerate(selected_scale_cols):
            res_col = orig_col if "Replace" in scaling_mode else created_or_modified_cols[i]
            comp_rows.append({
                "Feature (Original)": orig_col,
                "Orig Min": working_df[orig_col].min(),
                "Orig Max": working_df[orig_col].max(),
                "Orig Mean": working_df[orig_col].mean(),
                "Orig Std": working_df[orig_col].std(),
                "Scaled Col": res_col,
                "Scaled Min": new_df[res_col].min(),
                "Scaled Max": new_df[res_col].max(),
                "Scaled Mean": new_df[res_col].mean(),
                "Scaled Std": new_df[res_col].std()
            })
        st.dataframe(pd.DataFrame(comp_rows).set_index("Feature (Original)").style.format("{:.3f}"), use_container_width=True)

        # Visual comparison
        st.subheader("Distribution Comparison: Before vs After")
        sample_feature = selected_scale_cols[0]
        sample_scaled = sample_feature if "Replace" in scaling_mode else created_or_modified_cols[0]

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig1 = px.histogram(working_df, x=sample_feature, nbins=30, title=f"Before: {sample_feature} (Raw)", color_discrete_sequence=['#829B83'])
            fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)
        with col_g2:
            fig2 = px.histogram(new_df, x=sample_scaled, nbins=30, title=f"After: {sample_scaled} ({method_name})", color_discrete_sequence=['#244B3C'])
            fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        return new_df, audit_msg

    return None
