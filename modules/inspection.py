"""
MODULE 2: Dataset Inspection
Enables thorough structural inspection of the dataset using core Pandas methods:
head(), tail(), shape, dtypes, info(), and describe(), along with a single-column deep inspector.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io


def render_inspection(df: pd.DataFrame):
    """Renders the Dataset Inspection module."""

    st.markdown('<div class="section-header-box"><h2>Module 2: Dataset Inspection & Structural Understanding</h2><p>Examine raw structure, identify feature types, check memory usage, and generate descriptive statistical summaries.</p></div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("The current dataset is empty. Please upload or load a valid dataset.")
        return

    # Pedagogical Explanation Box
    with st.expander("Why Dataset Inspection is Critical in Machine Learning", expanded=True):
        st.markdown("""
        <div class="concept-box">
            <h4>Educational Foundations: Why inspect before modeling?</h4>
            <p>
            Machine learning algorithms cannot reason about semantic column meanings—they operate purely on numbers and mathematical vectors.
            Inspecting the dataset ensures we understand:
            </p>
            <ul>
                <li><b>Shape & Dimensions:</b> Determines whether sample size ($N$) is adequate relative to the number of features ($D$).</li>
                <li><b>Data Types (<code>dtypes</code>):</b> Identifies non-numeric features requiring encoding (strings, categories) or type-casting (e.g. numeric stored as object).</li>
                <li><b>Boundary Observations:</b> <code>head()</code> and <code>tail()</code> catch row misalignment, header formatting errors, and sentinel values (e.g., -999, "?", "NA").</li>
                <li><b>Statistical Distribution:</b> <code>describe()</code> exposes central tendencies, scale disparities, skewness, and extreme potential outliers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Tabs for organized view
    tab_preview, tab_schema, tab_stats, tab_inspector = st.tabs([
        "Data Preview (Head & Tail)",
        "Schema & Data Types",
        "Descriptive Statistics",
        "Single Column Inspector"
    ])

    # ----------------- TAB 1: Preview -----------------
    with tab_preview:
        st.subheader("Dataset Preview")
        col_n, col_shape = st.columns([1, 1])
        with col_n:
            n_rows = st.slider("Number of preview rows to display:", min_value=3, max_value=25, value=5, step=1)
        with col_shape:
            st.markdown(f"""
            **Current Dimensions:**
            - **Total Rows (Observations):** `{df.shape[0]:,}`
            - **Total Columns (Features):** `{df.shape[1]:,}`
            - **Total Cells:** `{df.shape[0] * df.shape[1]:,}`
            """)

        st.markdown(f"**First {n_rows} Records (`head({n_rows})`)**")
        st.dataframe(df.head(n_rows), use_container_width=True)

        st.markdown(f"**Last {n_rows} Records (`tail({n_rows})`)**")
        st.dataframe(df.tail(n_rows), use_container_width=True)

        # Simple explanation of head() and tail()
        st.markdown("""
        > **Pandas Concept Note:**  
        > - `df.head(n)`: Retrieves the initial $n$ rows from the DataFrame. Crucial for verifying successful file ingestion and column alignment.  
        > - `df.tail(n)`: Retrieves the final $n$ rows. Helps identify truncated files, summary rows, or trailing garbage characters often present in CSV exports.
        """)

    # ----------------- TAB 2: Schema & Types -----------------
    with tab_schema:
        st.subheader("Schema, Data Types & Info Summary")

        # Memory usage calculation
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()

        # Build structured schema table
        schema_data = []
        for col in df.columns:
            dtype_val = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            null_count = int(df[col].isna().sum())
            null_pct = (null_count / len(df)) * 100
            unique_count = int(df[col].nunique(dropna=True))
            sample_val = str(df[col].dropna().iloc[0]) if non_null > 0 else "N/A"

            # Categorize general type
            if pd.api.types.is_numeric_dtype(df[col]):
                gen_type = "Numerical (Continuous/Discrete)"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                gen_type = "Datetime"
            elif pd.api.types.is_bool_dtype(df[col]):
                gen_type = "Boolean"
            else:
                gen_type = "Categorical / Text"

            schema_data.append({
                "Column Name": col,
                "Inferred Category": gen_type,
                "Pandas dtype": dtype_val,
                "Non-Null Count": non_null,
                "Null Count": null_count,
                "Null Percentage (%)": f"{null_pct:.2f}%",
                "Unique Values": unique_count,
                "Example Value": sample_val
            })

        schema_df = pd.DataFrame(schema_data)
        st.dataframe(schema_df, use_container_width=True)

        # Separation of columns
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        other_cols = [c for c in df.columns if c not in num_cols and c not in cat_cols]

        col_num, col_cat = st.columns(2)
        with col_num:
            st.markdown(f"**Numerical Features ({len(num_cols)}):**")
            st.write(num_cols if num_cols else "None detected")
        with col_cat:
            st.markdown(f"**Categorical / String Features ({len(cat_cols)}):**")
            st.write(cat_cols if cat_cols else "None detected")

        if other_cols:
            st.markdown(f"**Other Features (Datetime / Boolean) ({len(other_cols)}):**")
            st.write(other_cols)

        with st.expander("Raw `df.info()` Console Output", expanded=False):
            st.text(info_str)

    # ----------------- TAB 3: Descriptive Stats -----------------
    with tab_stats:
        st.subheader("Descriptive Statistical Summaries (`describe()`)")

        stat_type = st.radio(
            "Select Feature Group for Descriptive Statistics:",
            ["Numerical Features", "Categorical Features", "All Features"],
            horizontal=True
        )

        try:
            if stat_type == "Numerical Features":
                num_df = df.select_dtypes(include=['number'])
                if num_df.empty:
                    st.info("No numerical features available in the current dataset.")
                else:
                    stats = num_df.describe().T
                    stats['median'] = num_df.median()
                    stats['skew'] = num_df.skew()
                    st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)
            elif stat_type == "Categorical Features":
                cat_df = df.select_dtypes(include=['object', 'category'])
                if cat_df.empty:
                    st.info("No categorical features available in the current dataset.")
                else:
                    st.dataframe(cat_df.describe().T, use_container_width=True)
            else:
                st.dataframe(df.describe(include='all').T, use_container_width=True)
        except Exception as e:
            st.error(f"Error computing descriptive statistics: {e}")

        st.markdown("""
        > **Statistical Metrics Guide:**  
        > - **Mean vs Median:** A large difference between mean and median indicates positive or negative skewness caused by extreme outliers.  
        > - **Standard Deviation (`std`):** Quantifies variance and feature dispersion. Features with large standard deviations dominate unscaled Euclidean distances.  
        > - **IQR (Q3 - Q1):** Interquartile Range covering the middle 50% of the sample, resistant to extreme values.
        """)

    # ----------------- TAB 4: Single Column Deep Dive -----------------
    with tab_inspector:
        st.subheader("Single Column Deep Inspector")
        selected_col = st.selectbox("Select a column to inspect in depth:", df.columns.tolist())

        if selected_col:
            col_series = df[selected_col]
            is_num = pd.api.types.is_numeric_dtype(col_series)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Data Type", str(col_series.dtype))
            with c2:
                st.metric("Missing Count", f"{col_series.isna().sum()} ({col_series.isna().mean()*100:.1f}%)")
            with c3:
                st.metric("Unique Count", f"{col_series.nunique()}")
            with c4:
                top_val = col_series.mode().iloc[0] if not col_series.mode().empty else "N/A"
                st.metric("Mode (Most Frequent)", str(top_val)[:15])

            col_chart, col_freq = st.columns([1.2, 0.8])

            with col_chart:
                if is_num:
                    st.markdown(f"**Histogram Distribution: `{selected_col}`**")
                    clean_data = col_series.dropna()
                    if not clean_data.empty:
                        fig = px.histogram(
                            clean_data,
                            x=clean_data,
                            nbins=25,
                            title=f"Distribution of {selected_col}",
                            color_discrete_sequence=['#244B3C'],
                            marginal="box"
                        )
                        fig.update_layout(xaxis_title=selected_col, yaxis_title="Frequency", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("All values in this numerical column are NaN.")
                else:
                    st.markdown(f"**Top Categories Frequency: `{selected_col}`**")
                    top_counts = col_series.value_counts(dropna=False).head(10).reset_index()
                    top_counts.columns = [selected_col, 'Count']
                    fig = px.bar(
                        top_counts,
                        x=selected_col,
                        y='Count',
                        title=f"Top 10 Categories in {selected_col}",
                        color='Count',
                        color_continuous_scale=[[0, '#D5DDD5'], [0.5, '#829B83'], [1.0, '#244B3C']]
                    )
                    fig.update_layout(xaxis_title=selected_col, yaxis_title="Count", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            with col_freq:
                st.markdown(f"**Frequency Breakdown (Top 10)**")
                val_counts = col_series.value_counts(dropna=False).head(10).to_frame(name="Count")
                val_counts["Percentage (%)"] = (val_counts["Count"] / len(col_series) * 100).map("{:.2f}%".format)
                st.dataframe(val_counts, use_container_width=True)

