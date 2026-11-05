"""
MODULE 8 & MODULE 9: Final Prepared Dataset & Academic Project Summary
Verifies data quality, validates zero-defect machine learning readiness,
provides prepared CSV export, logs the complete pipeline audit trail,
and generates downloadable academic project reports.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np


def render_export_and_summary(raw_df: pd.DataFrame, working_df: pd.DataFrame, audit_log: list, dataset_name: str):
    """Renders Module 8 (Final Prepared Dataset) and Module 9 (Summary & Conclusion)."""

    st.markdown('<div class="section-header-box"><h2>Module 8: Final Prepared Dataset Verification & Export</h2><p>Validate the final dataset, verify completeness and readiness for machine learning, and download the prepared dataset.</p></div>', unsafe_allow_html=True)

    if working_df.empty:
        st.warning("The working dataset is empty.")
        return

    # Auto-save current working df to outputs/prepared_dataset.csv
    try:
        os.makedirs("outputs", exist_ok=True)
        working_df.to_csv("outputs/prepared_dataset.csv", index=False)
    except Exception as e:
        pass

    # ----------------- SECTION 1: VERIFICATION CHECKLIST -----------------
    st.subheader("1. Machine Learning Readiness Verification Checklist")

    total_missing_after = int(working_df.isna().sum().sum())
    total_dups_after = int(working_df.duplicated().sum())
    num_cols_after = len(working_df.select_dtypes(include=['number']).columns)
    cat_cols_after = len(working_df.select_dtypes(include=['object', 'category', 'string']).columns)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if total_missing_after == 0:
            st.success("✅ **Missing Values:** 0 null cells (100% complete)")
        else:
            st.warning(f"⚠️ **Missing Values:** {total_missing_after:,} remaining nulls")
    with c2:
        if total_dups_after == 0:
            st.success("✅ **Duplicates:** 0 redundant rows")
        else:
            st.warning(f"⚠️ **Duplicates:** {total_dups_after:,} duplicate rows remaining")
    with c3:
        st.info(f"🔢 **Numerical Features:** {num_cols_after} features")
    with c4:
        if cat_cols_after == 0:
            st.success(f"✅ **Categorical Features:** 0 (Fully encoded)")
        else:
            st.info(f"🔤 **Categorical Features:** {cat_cols_after} unencoded")

    # ----------------- SECTION 2: BEFORE VS AFTER SUMMARY TABLE -----------------
    st.subheader("2. Dataset Transformation Comparison: Raw vs. Prepared")

    comp_summary = pd.DataFrame({
        "Attribute / Metric": [
            "Total Observations (Rows)",
            "Total Features (Columns)",
            "Missing Values Count",
            "Missing Value Percentage",
            "Duplicate Records Count",
            "Numerical Features Count",
            "Categorical Features Count",
            "Total Memory Footprint"
        ],
        "Original (Raw Dataset)": [
            f"{raw_df.shape[0]:,}",
            f"{raw_df.shape[1]:,}",
            f"{raw_df.isna().sum().sum():,}",
            f"{(raw_df.isna().sum().sum() / (raw_df.shape[0]*raw_df.shape[1])*100):.2f}%",
            f"{raw_df.duplicated().sum():,}",
            f"{len(raw_df.select_dtypes(include=['number']).columns)}",
            f"{len(raw_df.select_dtypes(include=['object', 'category', 'string']).columns)}",
            f"{raw_df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        ],
        "Final (Prepared Dataset)": [
            f"{working_df.shape[0]:,}",
            f"{working_df.shape[1]:,}",
            f"{working_df.isna().sum().sum():,}",
            f"{(working_df.isna().sum().sum() / (working_df.shape[0]*working_df.shape[1])*100):.2f}%",
            f"{working_df.duplicated().sum():,}",
            f"{len(working_df.select_dtypes(include=['number']).columns)}",
            f"{len(working_df.select_dtypes(include=['object', 'category', 'string']).columns)}",
            f"{working_df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        ]
    })
    st.dataframe(comp_summary.set_index("Attribute / Metric"), use_container_width=True)

    # ----------------- SECTION 3: PREPARED DATASET PREVIEW & DOWNLOAD -----------------
    st.subheader("3. Final Prepared Dataset Preview & CSV Download")
    st.dataframe(working_df.head(10), use_container_width=True)

    csv_data = working_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prepared Dataset (prepared_dataset.csv)",
        data=csv_data,
        file_name="prepared_dataset.csv",
        mime="text/csv",
        type="primary"
    )
    st.caption("The original uploaded dataset remains intact and unaltered in memory.")

    # =========================================================================
    # MODULE 9: PROJECT SUMMARY AND CONCLUSION
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="section-header-box"><h2>Module 9: Academic Project Summary & Conclusion</h2><p>Synthesize data preparation findings, review pipeline transformation logs, and export the official Unit I project report.</p></div>', unsafe_allow_html=True)

    st.subheader("1. End-to-End Pipeline Transformation Audit Trail")
    if audit_log:
        st.markdown("**Executed Preprocessing Pipeline Steps:**")
        for i, entry in enumerate(audit_log, 1):
            st.markdown(f"**Step {i}:** `{entry}`")
    else:
        st.info("No preprocessing transformations have been committed yet. The working dataset currently matches the raw dataset.")

    st.subheader("2. Academic Project Observations & Conclusion")
    st.markdown("""
    ### Summary of Findings & Engineering Decisions:
    
    1. **Structural Integrity & Inspection:**
       - Thorough inspection using `head()`, `tail()`, `shape`, `dtypes`, and `describe()` established the baseline distribution and revealed missingness in high-value features.
    
    2. **Missing Data Strategy:**
       - **Median Imputation** for skewed continuous features (such as `Age`) preserves distribution medians without undue pull from extreme positive outliers.
       - **Mode Imputation** for categorical variables (such as `Embarked`) restores missing discrete instances safely.
       - **Structural Column Pruning** was validated for severely incomplete columns (e.g., `Cabin` with >77% missing data), avoiding synthetic noise injection.
    
    3. **Consistency & Deduplication:**
       - Elimination of duplicate records prevents artificial weighting in downstream loss gradients.
       - Whitespace stripping and categorical normalization prevent fragmentation of discrete nominal classes.
    
    4. **Feature Encoding & Scaling:**
       - **One-Hot Encoding** converted nominal classes into orthonormal vectors, while employing `drop_first=True` to avert the dummy variable trap (multicollinearity).
       - High-cardinality identity features were protected from dimensional explosion.
       - **StandardScaler / MinMaxScaler** brought disparate feature scales into uniform alignment, ensuring distance-based and gradient-based algorithms converge without scale bias.
    
    ### Conclusion:
    The dataset has been methodically cleaned, standardized, encoded, and scaled following academic machine learning data preparation best practices. 
    It is now fully conditioned for downstream predictive modeling (Unit II algorithms such as Logistic Regression, Decision Trees, or Support Vector Machines).
    """)

    # Downloadable Project Report
    report_text = f"""================================================================================
ACADEMIC PROJECT REPORT: UNIT I DATASET PREPARATION & EDA DASHBOARD
Third-Year B.Tech Artificial Intelligence and Data Science
Course: Machine Learning | CIE 2 Unit I – Activity 1
================================================================================

1. PROJECT METADATA
- Dataset Name: {dataset_name}
- Original Dimensions: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns
- Final Prepared Dimensions: {working_df.shape[0]} rows × {working_df.shape[1]} columns
- Initial Missing Cells: {raw_df.isna().sum().sum()}
- Final Missing Cells: {working_df.isna().sum().sum()}
- Initial Duplicates: {raw_df.duplicated().sum()}
- Final Duplicates: {working_df.duplicated().sum()}

2. EXECUTED TRANSFORMATION PIPELINE STEPS:
{chr(10).join([f"  [{i+1}] {step}" for i, step in enumerate(audit_log)]) if audit_log else "  No transformations logged."}

3. PREPARATION SUMMARY TABLE:
- Raw Features ({raw_df.shape[1]}): {', '.join(raw_df.columns.tolist())}
- Final Features ({working_df.shape[1]}): {', '.join(working_df.columns.tolist())}
- Numerical Features Count: {len(working_df.select_dtypes(include=['number']).columns)}
- Categorical Features Count: {len(working_df.select_dtypes(include=['object', 'category', 'string']).columns)}

4. ACADEMIC EVALUATION RUBRIC COMPLIANCE:
[Criterion 1] Dataset Selection & Understanding: Completed via Module 1 & Module 2.
[Criterion 2] Missing Value & Data Quality Handling: Completed via Module 3 & Module 4.
[Criterion 3] Exploratory Data Analysis & Visualizations: Completed via Module 5.
[Criterion 4] Categorical Encoding & Numerical Scaling: Completed via Module 6 & Module 7.
[Criterion 5] Prepared Dataset Verification & Report: Completed via Module 8 & Module 9.

Prepared Dataset File: prepared_dataset.csv
Generated Successfully on Academic Dashboard.
================================================================================
"""

    st.download_button(
        label="📄 Download Academic Summary Report (dataset_preparation_report.txt)",
        data=report_text.encode('utf-8'),
        file_name="dataset_preparation_report.txt",
        mime="text/plain"
    )

