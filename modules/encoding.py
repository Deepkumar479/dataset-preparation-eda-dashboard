"""
MODULE 6: Categorical Encoding
Transforms non-numeric features into numerical representations suitable for ML algorithms.
Supports One-Hot Encoding (with drop_first) and Ordinal Encoding, while protecting unique identifiers.
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def render_encoding(working_df: pd.DataFrame):
    """
    Renders Categorical Encoding module.
    Returns: (updated_df, audit_message) if changes applied, else None.
    """
    st.markdown('<div class="section-header-box"><h2>Module 6: Categorical Feature Encoding</h2><p>Convert discrete text and categorical attributes into mathematical representations for machine learning matrices.</p></div>', unsafe_allow_html=True)

    if working_df.empty:
        st.warning("The active dataset is empty.")
        return None

    # Academic Theory Box
    with st.expander("Academic Guide: Why Machine Learning Requires Numerical Encoding", expanded=True):
        st.markdown(r"""
        <div class="concept-box">
            <h4>Mathematical Foundations of Categorical Encoding</h4>
            <p>
            Machine learning models are mathematical systems executing matrix multiplications, dot products, and gradient updates:
            $$\\hat{y} = \\mathbf{w}^T \\mathbf{x} + b$$
            String categories such as <code>"male"</code> or <code>"female"</code> cannot undergo algebraic operations. Therefore:
            </p>
            <ul>
                <li><b>One-Hot Encoding (OHE):</b> Converts each unique category into a binary indicator column ($0$ or $1$). 
                Essential for <i>nominal</i> variables where categories possess no intrinsic hierarchical rank (e.g., <code>Embarked: 'C', 'Q', 'S'</code> or <code>Sex: 'male', 'female'</code>).</li>
                <li><b>The Dummy Variable Trap:</b> When all $k$ categories are represented with $k$ binary columns alongside a bias intercept $b$, perfect collinearity ensues (the sum of columns equals $1$). Setting <code>drop_first=True</code> (using $k-1$ columns) prevents matrix singularity.</li>
                <li><b>Ordinal Encoding:</b> Assigns integer ranks ($0, 1, 2, \\dots$) to categories possessing an explicit natural ordering (e.g., Education levels, or <code>Pclass: 1st > 2nd > 3rd</code>). Applying ordinal encoding to nominal data falsely injects arbitrary distance metrics!</li>
                <li><b>High-Cardinality Identifiers:</b> Features with unique values per record (e.g., <code>PassengerId</code>, <code>Name</code>, <code>Ticket</code>) must <b>never</b> be one-hot encoded, as they cause explosive dimensionality and severe overfitting without generalizable signal.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Detect categorical candidates
    cat_cols = working_df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # Identify potential high-cardinality identifiers to exclude/warn
    identifier_cols = []
    eligible_cat_cols = []
    for c in cat_cols:
        unique_ratio = working_df[c].nunique() / len(working_df)
        if unique_ratio > 0.6 or working_df[c].nunique() > 50:
            identifier_cols.append(c)
        else:
            eligible_cat_cols.append(c)

    col_info1, col_info2 = st.columns([1, 1])
    with col_info1:
        st.markdown(f"**Eligible Categorical Features ({len(eligible_cat_cols)}):**")
        st.write(eligible_cat_cols if eligible_cat_cols else "None detected")
    with col_info2:
        st.markdown(f"**Identified High-Cardinality / ID Columns ({len(identifier_cols)}):**")
        st.write(identifier_cols if identifier_cols else "None detected")
        if identifier_cols:
            st.caption("⚠️ High-cardinality strings like Name or Ticket are excluded from default encoding to prevent the curse of dimensionality.")

    if not eligible_cat_cols and not cat_cols:
        st.success("All features in the working dataset are already numerical! No categorical encoding needed.")
        return None

    st.markdown("---")
    st.subheader("Interactive Encoding Configuration")

    encoding_type = st.radio(
        "Select Encoding Technique:",
        ["One-Hot Encoding (Nominal Features)", "Ordinal Encoding (Ranked Features)"],
        horizontal=True
    )

    if "One-Hot Encoding" in encoding_type:
        st.markdown("#### Configure One-Hot Encoding")

        selected_ohe_cols = st.multiselect(
            "Select Categorical Columns to One-Hot Encode:",
            options=eligible_cat_cols if eligible_cat_cols else cat_cols,
            default=eligible_cat_cols[:2] if len(eligible_cat_cols) >= 2 else eligible_cat_cols
        )

        drop_first = st.checkbox(
            "Apply `drop_first=True` (Drop first dummy column to prevent multicollinearity)",
            value=True,
            help="Recommended in machine learning to avoid the dummy variable trap in linear and logistic regression."
        )

        if not selected_ohe_cols:
            st.info("Select at least one column to apply One-Hot Encoding.")
        else:
            # Preview of transformation
            st.markdown("**Preview of Categories to be Encoded:**")
            cat_summary = []
            for col in selected_ohe_cols:
                cats = working_df[col].dropna().unique().tolist()
                new_col_count = len(cats) - 1 if drop_first else len(cats)
                cat_summary.append({
                    "Feature": col,
                    "Unique Categories": len(cats),
                    "Distinct Values": str(cats)[:60],
                    "New Columns Created": new_col_count
                })
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

            if st.button("Apply One-Hot Encoding", type="primary"):
                shape_before = working_df.shape
                new_df = pd.get_dummies(working_df, columns=selected_ohe_cols, drop_first=drop_first, dtype=int)
                shape_after = new_df.shape

                # Identify created columns
                created_cols = [c for c in new_df.columns if c not in working_df.columns]

                audit_msg = (
                    f"Applied One-Hot Encoding on {selected_ohe_cols} (drop_first={drop_first}). "
                    f"Shape changed from {shape_before} to {shape_after}. Created {len(created_cols)} new dummy features: {created_cols}."
                )

                st.success(f"**Encoding Successful!** Dataset shape expanded from `{shape_before}` to `{shape_after}`.")
                st.markdown(f"**Newly Generated Binary Columns ({len(created_cols)}):** `{created_cols}`")
                st.dataframe(new_df.head(4), use_container_width=True)

                return new_df, audit_msg

    else:
        st.markdown("#### Configure Ordinal Encoding")

        col_ord_target = st.selectbox("Select Feature for Ordinal Encoding:", eligible_cat_cols if eligible_cat_cols else cat_cols)

        if col_ord_target:
            unique_ord_cats = working_df[col_ord_target].dropna().astype(str).unique().tolist()
            st.markdown(f"**Identified Unique Categories in `{col_ord_target}`:**")
            st.write(unique_ord_cats)

            user_order_str = st.text_input(
                "Specify ordered sequence (comma-separated, lowest rank to highest rank):",
                value=", ".join([str(x) for x in sorted(unique_ord_cats)])
            )

            if st.button("Apply Ordinal Encoding", type="primary"):
                try:
                    category_order = [x.strip() for x in user_order_str.split(",")]
                    mapping_dict = {cat: idx for idx, cat in enumerate(category_order)}

                    new_df = working_df.copy()
                    encoded_col_name = f"{col_ord_target}_ordinal"
                    new_df[encoded_col_name] = new_df[col_ord_target].astype(str).map(mapping_dict)

                    audit_msg = (
                        f"Applied Ordinal Encoding on '{col_ord_target}' -> '{encoded_col_name}' "
                        f"with rank mapping: {mapping_dict}."
                    )

                    st.success(f"**Ordinal Encoding Applied!** New column `{encoded_col_name}` created.")
                    st.write(mapping_dict)
                    st.dataframe(new_df[[col_ord_target, encoded_col_name]].head(5), use_container_width=True)

                    return new_df, audit_msg
                except Exception as e:
                    st.error(f"Error applying ordinal encoding: {e}")

    return None
