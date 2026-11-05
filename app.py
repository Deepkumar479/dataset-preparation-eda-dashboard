"""
=============================================================================
DATASET PREPARATION AND EXPLORATORY DATA ANALYSIS (EDA) DASHBOARD
An Interactive System for Data Inspection, Cleaning, Analysis & Preprocessing
Unit I Machine Learning - CIE 2 Activity 1
B.Tech Artificial Intelligence and Data Science
=============================================================================
"""

import os
import streamlit as st
import pandas as pd

# Import modular application components
from modules.overview import render_overview
from modules.inspection import render_inspection
from modules.missing_values import render_missing_values
from modules.data_cleaning import render_data_cleaning
from modules.eda import render_eda
from modules.encoding import render_encoding
from modules.scaling import render_scaling
from modules.export import render_export_and_summary


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Dataset Prep & EDA Dashboard | ML Unit I",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External Styling
def load_css(css_file_path: str):
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")


# Default Titanic Dataset Loader
@st.cache_data
def load_default_dataset():
    sample_path = os.path.join("data", "sample_dataset.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    else:
        # Fallback inline generation if file not found
        import urllib.request
        os.makedirs("data", exist_ok=True)
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        urllib.request.urlretrieve(url, sample_path)
        return pd.read_csv(sample_path)


# Initialize Session State
if "raw_df" not in st.session_state:
    try:
        default_data = load_default_dataset()
        st.session_state.raw_df = default_data.copy()
        st.session_state.working_df = default_data.copy()
        st.session_state.dataset_name = "Titanic - Machine Learning from Disaster"
        st.session_state.source_info = "https://www.kaggle.com/datasets/yasserh/titanic-dataset"
        st.session_state.audit_log = []
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        st.session_state.raw_df = pd.DataFrame()
        st.session_state.working_df = pd.DataFrame()
        st.session_state.dataset_name = "Empty Dataset"
        st.session_state.source_info = "Local"
        st.session_state.audit_log = []


# =============================================================================
# SIDEBAR NAVIGATION & DATASET SELECTION
# =============================================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 16px 0;">
    <h3 style="margin: 0; color: #244B3C; font-size: 19px; font-weight: 700;">ML Preparation Hub</h3>
    <p style="margin: 3px 0 0 0; color: #829B83; font-size: 12px; font-weight: 500;">Unit I: Dataset Preparation & EDA</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Dataset Source Selection
st.sidebar.markdown("### 📁 Dataset Selector")
dataset_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Titanic (Kaggle Demonstration)", "Upload Custom CSV Dataset"],
    index=0
)

if dataset_source == "Upload Custom CSV Dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Tabular CSV File",
        type=["csv"],
        help="Upload any standard comma-separated tabular dataset (.csv)"
    )

    if uploaded_file is not None:
        try:
            # Handle encoding fallbacks
            try:
                uploaded_df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                uploaded_df = pd.read_csv(uploaded_file, encoding='latin1')

            if uploaded_df.empty:
                st.sidebar.error("Uploaded CSV file is empty!")
            else:
                # Check if this is a newly uploaded file
                if st.session_state.dataset_name != uploaded_file.name:
                    st.session_state.raw_df = uploaded_df.copy()
                    st.session_state.working_df = uploaded_df.copy()
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.source_info = f"Uploaded File: {uploaded_file.name}"
                    st.session_state.audit_log = []
                    st.sidebar.success(f"Loaded: `{uploaded_file.name}` ({len(uploaded_df)} rows)")
        except Exception as err:
            st.sidebar.error(f"Error parsing CSV: {err}")
else:
    # Check if we should switch back to Titanic
    if st.session_state.dataset_name != "Titanic - Machine Learning from Disaster":
        default_data = load_default_dataset()
        st.session_state.raw_df = default_data.copy()
        st.session_state.working_df = default_data.copy()
        st.session_state.dataset_name = "Titanic - Machine Learning from Disaster"
        st.session_state.source_info = "https://www.kaggle.com/datasets/yasserh/titanic-dataset"
        st.session_state.audit_log = []

st.sidebar.markdown("---")

# Module Navigation
st.sidebar.markdown("### 🧭 Application Modules")
modules_list = [
    "Module 1: Dashboard Overview",
    "Module 2: Dataset Inspection",
    "Module 3: Missing Value Analysis",
    "Module 4: Duplicate & Data Consistency",
    "Module 5: Exploratory Data Analysis (EDA)",
    "Module 6: Categorical Encoding",
    "Module 7: Numerical Feature Scaling",
    "Module 8: Final Prepared Dataset",
    "Module 9: Summary & Conclusion"
]

selected_module = st.sidebar.radio(
    "Select Module to Explore:",
    modules_list,
    index=0
)

st.sidebar.markdown("---")

# Active Dataset Status Card in Sidebar
working_df = st.session_state.working_df
raw_df = st.session_state.raw_df

st.sidebar.markdown("### 📊 Active Working Data Status")
st.sidebar.markdown(f"""
- **Active Dataset:** `{st.session_state.dataset_name[:20]}`
- **Dimensions:** `{working_df.shape[0]}` rows × `{working_df.shape[1]}` cols
- **Missing Cells:** `{int(working_df.isna().sum().sum())}`
- **Duplicate Rows:** `{int(working_df.duplicated().sum())}`
- **Pipeline Actions Logged:** `{len(st.session_state.audit_log)}`
""")

# Reset Pipeline Button
if st.sidebar.button("🔄 Reset Working Dataset", help="Reverts working data back to the original unmodified raw dataset"):
    st.session_state.working_df = st.session_state.raw_df.copy()
    st.session_state.audit_log = []
    st.sidebar.success("Working dataset reset to original state!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11px; color: #556658; line-height: 1.4;">
    <b>Academic Project:</b> CIE 2 Unit I<br>
    <b>Course:</b> Machine Learning<br>
    <b>Class:</b> T.Y. B.Tech AI & Data Science<br>
    <b>Theme:</b> Warm Ivory + Forest Green
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN MODULE ROUTING & EXECUTION
# =============================================================================

if selected_module == "Module 1: Dashboard Overview":
    render_overview(
        raw_df=st.session_state.raw_df,
        working_df=st.session_state.working_df,
        dataset_name=st.session_state.dataset_name,
        source_info=st.session_state.source_info
    )

elif selected_module == "Module 2: Dataset Inspection":
    render_inspection(st.session_state.working_df)

elif selected_module == "Module 3: Missing Value Analysis":
    result = render_missing_values(st.session_state.working_df)
    if result is not None:
        new_df, audit_entry = result
        st.session_state.working_df = new_df
        st.session_state.audit_log.append(audit_entry)
        st.rerun()

elif selected_module == "Module 4: Duplicate & Data Consistency":
    result = render_data_cleaning(st.session_state.working_df)
    if result is not None:
        new_df, audit_entry = result
        st.session_state.working_df = new_df
        st.session_state.audit_log.append(audit_entry)
        st.rerun()

elif selected_module == "Module 5: Exploratory Data Analysis (EDA)":
    render_eda(st.session_state.working_df)

elif selected_module == "Module 6: Categorical Encoding":
    result = render_encoding(st.session_state.working_df)
    if result is not None:
        new_df, audit_entry = result
        st.session_state.working_df = new_df
        st.session_state.audit_log.append(audit_entry)
        st.rerun()

elif selected_module == "Module 7: Numerical Feature Scaling":
    result = render_scaling(st.session_state.working_df)
    if result is not None:
        new_df, audit_entry = result
        st.session_state.working_df = new_df
        st.session_state.audit_log.append(audit_entry)
        st.rerun()

elif selected_module in ["Module 8: Final Prepared Dataset", "Module 9: Summary & Conclusion"]:
    render_export_and_summary(
        raw_df=st.session_state.raw_df,
        working_df=st.session_state.working_df,
        audit_log=st.session_state.audit_log,
        dataset_name=st.session_state.dataset_name
    )

