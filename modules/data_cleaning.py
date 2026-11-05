"""
MODULE 4: Duplicate and Data Consistency
Audits duplicate rows, identifies leading/trailing whitespace inconsistencies in categorical data,
and provides interactive remediation tools to ensure semantic consistency.
"""

import streamlit as st
import pandas as pd


def render_data_cleaning(working_df: pd.DataFrame):
    """
    Renders Duplicate Records and Data Consistency module.
    Returns: (updated_df, audit_message) if changes applied, else None.
    """
    st.markdown('<div class="section-header-box"><h2>Module 4: Duplicate Records & Data Consistency</h2><p>Detect redundant records, audit string inconsistencies, and standardize categorical representations.</p></div>', unsafe_allow_html=True)

    if working_df.empty:
        st.warning("The active dataset is empty.")
        return None

    # Educational Concept Box
    with st.expander("Academic Guide: Why Duplicate & Consistency Cleaning Matters", expanded=True):
        st.markdown("""
        <div class="concept-box">
            <h4>Data Hygiene & Structural Consistency</h4>
            <p>
            Data inconsistencies compromise algorithmic learning in subtle ways:
            </p>
            <ul>
                <li><b>Duplicate Rows:</b> Identical records artificially amplify the weight of specific instances, biasing loss functions and leading to overfitted models.</li>
                <li><b>Whitespace Inconsistencies:</b> Python distinguishes <code>"female"</code>, <code>" female"</code>, and <code>"female "</code> as distinct classes. When encoded, this generates spurious extra columns.</li>
                <li><b>Case Sensitivity:</b> Nominal classes recorded in mixed case (e.g., <code>"Male"</code> vs <code>"male"</code>) fracture single categories into fragments.</li>
                <li><b>Conservative Editing:</b> Unusual or outlier values should not be automatically eliminated without domain justification.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    tab_dups, tab_spaces, tab_casing = st.tabs([
        "Duplicate Rows Audit",
        "Whitespace Inconsistency Audit",
        "Categorical Label Standardization"
    ])

    # ----------------- TAB 1: DUPLICATES -----------------
    with tab_dups:
        st.subheader("Duplicate Record Detection")

        dup_mask = working_df.duplicated(keep=False)
        dup_count = int(working_df.duplicated().sum())
        total_dup_rows = int(dup_mask.sum())

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            if dup_count == 0:
                st.success("No duplicate records detected. Every row in the active dataset is structurally unique!")
            else:
                st.warning(f"**Duplicates Detected:** Found {dup_count:,} duplicate rows ({total_dup_rows:,} total matching records).")

        with col_d2:
            st.metric("Duplicate Count", f"{dup_count:,}", help="Number of rows that are exact duplicates of an earlier row")

        if dup_count > 0:
            st.markdown("**Duplicate Records Preview:**")
            dup_records = working_df[dup_mask].sort_values(by=working_df.columns.tolist())
            st.dataframe(dup_records, use_container_width=True)

            if st.button("Remove Duplicate Records", type="primary"):
                rows_before = len(working_df)
                new_df = working_df.drop_duplicates(keep='first').copy()
                rows_after = len(new_df)
                removed = rows_before - rows_after
                audit_msg = f"Removed {removed} duplicate rows. Dataset row count reduced from {rows_before} to {rows_after}."
                st.success(audit_msg)
                return new_df, audit_msg
        else:
            st.info("No row deduplication needed. The dataset integrity is verified.")

    # ----------------- TAB 2: WHITESPACE AUDIT -----------------
    with tab_spaces:
        st.subheader("Whitespace & Padding Inconsistency Audit")

        cat_cols = working_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        if not cat_cols:
            st.info("No categorical/string columns present in the current dataset to audit for whitespace.")
        else:
            whitespace_issues = []

            for col in cat_cols:
                series_str = working_df[col].dropna().astype(str)
                # Check leading or trailing whitespaces
                has_padded = series_str.apply(lambda x: len(x) != len(x.strip())).sum()
                if has_padded > 0:
                    whitespace_issues.append({
                        "Column": col,
                        "Inconsistent Padded Rows": int(has_padded),
                        "Percentage (%)": f"{(has_padded / len(working_df)) * 100:.2f}%",
                        "Example Offender": series_str[series_str.apply(lambda x: len(x) != len(x.strip()))].iloc[0]
                    })

            if whitespace_issues:
                st.warning(f"Found whitespace padding issues in {len(whitespace_issues)} column(s).")
                ws_df = pd.DataFrame(whitespace_issues)
                st.dataframe(ws_df, use_container_width=True)

                cols_to_strip = st.multiselect(
                    "Select Columns to Strip Whitespace From:",
                    options=[item["Column"] for item in whitespace_issues],
                    default=[item["Column"] for item in whitespace_issues]
                )

                if st.button("Trim Leading & Trailing Whitespaces", type="primary"):
                    new_df = working_df.copy()
                    for col in cols_to_strip:
                        new_df[col] = new_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

                    audit_msg = f"Trimmed leading/trailing whitespaces in column(s): {', '.join(cols_to_strip)}."
                    st.success(audit_msg)
                    return new_df, audit_msg
            else:
                st.success("No leading or trailing whitespaces detected in any categorical features!")
                st.info("All string representations have clean character boundaries.")

    # ----------------- TAB 3: CATEGORICAL CASING -----------------
    with tab_casing:
        st.subheader("Categorical Casing Standardization")

        if not cat_cols:
            st.info("No categorical features available to inspect casing.")
        else:
            col_target_cat = st.selectbox("Inspect Category Labels for Column:", cat_cols, key="casing_col_select")

            if col_target_cat:
                raw_cats = working_df[col_target_cat].dropna().astype(str).unique().tolist()
                st.markdown(f"**Current Unique Categories in `{col_target_cat}` ({len(raw_cats)}):**")
                st.write(raw_cats)

                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    casing_choice = st.selectbox(
                        "Choose Casing Format to Apply:",
                        ["Standardize to Title Case (e.g. 'Female')", "Standardize to Lowercase (e.g. 'female')", "Standardize to Uppercase (e.g. 'FEMALE')"]
                    )
                with col_c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Apply Casing Standardization", type="primary"):
                        new_df = working_df.copy()
                        if "Lowercase" in casing_choice:
                            new_df[col_target_cat] = new_df[col_target_cat].apply(lambda x: x.lower() if isinstance(x, str) else x)
                            audit_msg = f"Standardized values in '{col_target_cat}' to lowercase."
                        elif "Uppercase" in casing_choice:
                            new_df[col_target_cat] = new_df[col_target_cat].apply(lambda x: x.upper() if isinstance(x, str) else x)
                            audit_msg = f"Standardized values in '{col_target_cat}' to uppercase."
                        else:
                            new_df[col_target_cat] = new_df[col_target_cat].apply(lambda x: x.title() if isinstance(x, str) else x)
                            audit_msg = f"Standardized values in '{col_target_cat}' to title case."

                        st.success(audit_msg)
                        return new_df, audit_msg

    return None

