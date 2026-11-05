# Dataset Preparation and Exploratory Data Analysis (EDA) Dashboard
**An Interactive System for Data Inspection, Cleaning, Analysis, and Preprocessing for Machine Learning**

*Third-Year B.Tech Artificial Intelligence and Data Science*  
*Course: Machine Learning | CIE 2: Unit I – Activity 1 (Dataset Preparation Activity)*

---

## 1. Project Introduction

In classical and modern machine learning lifecycles, real-world raw datasets cannot be fed directly into mathematical learning algorithms. Raw tabular data commonly suffers from missing values, duplicate records, non-standard text casing, varying feature scales, and non-numeric categorical attributes. Feeding unconditioned data into algorithms results in biased estimators, matrix singularities, non-converging gradient descent routines, and degraded generalization performance.

This project delivers a **full-featured, interactive, web-based analytics and data preparation dashboard** engineered with **Streamlit**, **Pandas**, **NumPy**, **Plotly**, **Seaborn**, and **Scikit-learn**. It empowers users and students to upload tabular datasets, diagnose data-quality anomalies, perform in-depth statistical exploratory data analysis, apply statistically sound missing-value imputations, encode categorical features, scale numerical attributes, and export a 100% machine-learning-ready prepared dataset (`prepared_dataset.csv`).

> **Unit I Academic Boundary:**  
> This project strictly focuses on dataset understanding, data cleaning, exploratory data analysis, and preprocessing. In alignment with Unit I curriculum constraints, **no predictive machine learning models** (such as Linear Regression, Logistic Regression, Decision Trees, or Random Forests) are implemented.

---

## 2. Problem Statement

Machine learning practitioners spend up to 80% of project time inspecting, cleaning, and preparing raw datasets. In an academic learning context, students often treat data preprocessing as a black-box script without understanding *why* certain imputations, encodings, or scaling transformations are chosen.

The objective of this activity is to build an interactive graphical dashboard that:
1. Replaces opaque scripts with visual inspection and interactive decision-making.
2. Demonstrates the mathematical and statistical rationale behind every preprocessing decision.
3. Preserves the original raw dataset intact in memory while tracking an immutable, chronological transformation audit trail.
4. Generates a zero-defect prepared dataset suitable for downstream machine learning algorithms in Unit II.

---

## 3. Technology Stack

- **Application Interface:** [Streamlit](https://streamlit.io/) (v1.35+)
- **Data Manipulation & Indexing:** [Pandas](https://pandas.pydata.org/) (v2.0+)
- **Numerical Operations & Vector Algebra:** [NumPy](https://numpy.org/) (v1.24+)
- **Interactive Data Visualisation:** [Plotly Express & Graph Objects](https://plotly.com/python/) (v5.15+)
- **Statistical Graphics:** [Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/)
- **Feature Preprocessing & Scaling:** [Scikit-learn](https://scikit-learn.org/) (v1.3+)
- **Language & Runtime:** Python 3.10+ / 3.11+ / 3.12+ / 3.14+

---

## 4. Dataset Information & Source

### Demonstration Dataset: Titanic – Machine Learning from Disaster
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) / [DataScienceDojo Mirror](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Local Location:** `data/sample_dataset.csv`
- **Initial Dimensions:** 891 rows × 12 columns
- **Feature Breakdown:**
  - `PassengerId` *(int64)*: Unique integer identifier per passenger (excluded from modeling).
  - `Survived` *(int64)*: Ground truth survival indicator ($0 = \text{No}, 1 = \text{Yes}$).
  - `Pclass` *(int64)*: Socioeconomic ticket class ($1 = 1^{\text{st}}, 2 = 2^{\text{nd}}, 3 = 3^{\text{rd}}$).
  - `Name` *(object)*: Full passenger name and social honorific title.
  - `Sex` *(object)*: Passenger biological gender (`male`, `female`).
  - `Age` *(float64)*: Fractional or integer passenger age (177 missing values, right-skewed).
  - `SibSp` *(int64)*: Number of siblings / spouses aboard.
  - `Parch` *(int64)*: Number of parents / children aboard.
  - `Ticket` *(object)*: Alphanumeric ticket reservation string.
  - `Fare` *(float64)*: Passenger ticket fare (continuous, range \$0.00 to \$512.33, heavily right-skewed).
  - `Cabin` *(object)*: Cabin allocation identifier (687 missing values, 77.10% missingness).
  - `Embarked` *(object)*: Port of embarkation (`C = Cherbourg`, `Q = Queenstown`, `S = Southampton`; 2 missing values).

### Custom Dataset Flexibility
The dashboard supports standard comma-separated tabular files (`.csv`). Users can upload any custom dataset via the sidebar to perform automated inspection, cleaning, EDA, encoding, and scaling.

---

## 5. Project Architecture & Directory Structure

```
d:\T.Y_Zeal\ML\Activities\Activiry01/
│
├── app.py                      # Main Streamlit application entry point & routing
├── dataset_preparation_eda.ipynb # Full academic Jupyter Notebook covering all 10 steps
├── requirements.txt            # Python dependencies specification
├── README.md                   # Comprehensive project documentation
├── REPORT_OUTLINE.md           # Formal academic submission report outline
├── test_app.py                 # Automated unit and module validation test suite
│
├── assets/
│   └── style.css               # Professional Warm Ivory & Forest Green theme custom CSS
│
├── data/
│   └── sample_dataset.csv      # Authentic Kaggle Titanic demonstration dataset
│
├── modules/
│   ├── __init__.py             # Python package initializer
│   ├── overview.py             # Module 1: Dashboard Overview & KPI Metrics
│   ├── inspection.py           # Module 2: Dataset Inspection & Structural dtypes
│   ├── missing_values.py       # Module 3: Missing Value Analysis & Imputation
│   ├── data_cleaning.py        # Module 4: Deduplication & Consistency Tools
│   ├── eda.py                  # Module 5: Exploratory Data Analysis & Plotly Visuals
│   ├── encoding.py             # Module 6: One-Hot & Ordinal Encoding
│   ├── scaling.py              # Module 7: StandardScaler & MinMaxScaler
│   └── export.py               # Modules 8 & 9: Verification, Export & Audit Report
│
└── outputs/
    └── prepared_dataset.csv    # Final 100% prepared machine-learning-ready CSV
```

---

## 6. Detailed Explanation of Dashboard Modules

### Module 1: Dashboard Overview
- Displays project title, subtitle, student course metadata, and academic evaluation badges.
- Displays 6 real-time metric cards:
  - Total Records (rows)
  - Total Columns (features)
  - Numerical Feature Count
  - Categorical Feature Count
  - Total Missing Values
  - Duplicate Records Count
- Provides dataset selector (Titanic built-in vs. custom CSV upload).
- Summarizes the 5-mark college evaluation scheme.

### Module 2: Dataset Inspection
- **Data Preview:** First $N$ rows via `head()` and last $N$ rows via `tail()` with an interactive row count slider.
- **Structural Schema:** Column names, Pandas `dtypes`, inferred semantic categories, non-null counts, null percentages, and unique value counts.
- **Descriptive Statistics:** Numerical features summary (mean, median, std dev, min, max, IQR, skewness) and categorical summaries (unique count, mode, top frequency).
- **Single Column Inspector:** Dropdown to isolate and inspect any individual column with an automated mini-histogram or bar chart.
- **Educational Callouts:** Explains why `head()`, `tail()`, `shape`, `dtypes`, `info()`, and `describe()` are essential initial steps.

### Module 3: Missing Value Analysis & Handling
- **Audit Table:** Computes exact null counts and percentages for every column.
- **Visualizations:**
  - Interactive Plotly bar chart displaying missingness percentages.
  - Missingness matrix heatmap visualizing row-by-column missing patterns.
- **Imputation Engine:**
  - *Numerical:* Median Imputation (robust against skewness and outliers) or Mean Imputation (for normal distributions).
  - *Categorical:* Mode Imputation (restores most frequent class).
  - *Structural:* Dropping columns with severe missingness (e.g. `Cabin` with 77% nulls) or dropping rows containing NaNs.
- **Academic Justification:** Educates students on MCAR, MAR, and MNAR mechanisms and warns against blanket row deletion.

### Module 4: Duplicate & Data Consistency
- **Duplicate Audit:** Scans for exact identical rows across all features and provides a preview of duplicate entries.
- **Deduplication Tool:** Removes duplicate records while preserving the first occurrence (`drop_duplicates(keep='first')`).
- **Whitespace Audit:** Scans categorical strings for leading or trailing whitespace padding (e.g. `" male "` vs `"male"`).
- **Whitespace Stripping:** Trims extraneous whitespace characters across selected categorical columns.
- **Casing Normalization:** Allows standardizing categories to Title Case, Lowercase, or Uppercase to eliminate casing discrepancies.

### Module 5: Exploratory Data Analysis (EDA)
- **Descriptive Profiles:** Generates complete numerical metrics (Q1, Median, Q3, IQR, Skewness, Mode) and categorical frequency tables.
- **5 Core Interactive Plotly Visualizations:**
  1. *Histogram with Marginal Rug Plot:* Inspects univariate numerical distributions with bin sliders.
  2. *Categorical Bar Chart:* Displays category frequencies with dynamic percentage labels.
  3. *Box & Violin Plots:* Exposes outliers beyond $1.5 \times \text{IQR}$, with optional category grouping (e.g. `Fare` grouped by `Pclass`).
  4. *Bivariate Scatter Plot:* Reveals multi-attribute interactions (e.g. `Age` vs. `Fare` colored by `Survived`).
  5. *Pearson Correlation Heatmap:* Color-coded matrix showing linear correlation coefficients across numerical attributes.
- **Authentic Observations:** Dynamically computes genuine observations (e.g., female survival rate of 74.2% vs. male survival rate of 18.9%, Fare right-skewness of +4.79).

### Module 6: Categorical Encoding
- **Pedagogical Foundations:** Explains why ML algorithms require numerical matrix inputs ($\mathbf{w}^T \mathbf{x} + b$).
- **One-Hot Encoding (OHE):** Converts nominal categories into orthogonal binary columns with `drop_first=True` to avert the dummy variable trap (multicollinearity).
- **Ordinal Encoding:** Applies ordered rank mappings ($0, 1, 2, \dots$) to categories with natural hierarchies.
- **Safety Safeguards:** Automatically detects and excludes high-cardinality identifiers (like `PassengerId`, `Name`, `Ticket`) to avoid the curse of dimensionality.
- **Transformation History:** Displays dataset shape before and after encoding, listing newly created features.

### Module 7: Numerical Feature Scaling
- **Mathematical Rationale:** Explains how disparate feature ranges distort Euclidean distances ($\sqrt{\sum (p_i - q_i)^2}$) in distance-based algorithms (KNN, SVM, K-Means) and warp gradient descent contours.
- **StandardScaler (Standardisation):** Centers data to $\mu = 0, \sigma = 1$ ($z = \frac{x - \mu}{\sigma}$).
- **MinMaxScaler (Normalisation):** Bounds values strictly within $[0, 1]$ ($x' = \frac{x - x_{min}}{x_{max} - x_{min}}$).
- **Leakage Warning:** Explicitly clarifies that in predictive workflows, scalers must be fit strictly on training splits.
- **Before-and-After Comparison:** Displays a side-by-side statistical table and dual histograms showing original vs. scaled distributions.

### Module 8: Final Prepared Dataset
- **Readiness Verification Checklist:** Verifies zero remaining missing values, zero duplicates, and numeric feature compatibility.
- **Comparison Table:** Side-by-side tabular comparison of original raw dataset vs. final prepared dataset (dimensions, missing count, memory).
- **CSV Export:** Download button for `prepared_dataset.csv`.
- **Memory Safety:** Guarantees that the original uploaded dataset remains immutable.

### Module 9: Summary & Conclusion
- **Chronological Audit Trail:** Displays a numbered log of every preprocessing step executed during the session.
- **Academic Synthesis:** Formulates conclusions on missing-data handling, deduplication, encoding, and scaling.
- **Downloadable Academic Report:** Generates a downloadable `.txt` report (`dataset_preparation_report.txt`) summarizing the preparation pipeline for faculty review.

---

## 7. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14 installed on your system.
- Git or direct folder access.

### Step 1: Open Terminal in Project Directory
```powershell
cd d:\T.Y_Zeal\ML\Activities\Activiry01
```

### Step 2: Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run the Automated Validation Suite
```powershell
python test_app.py
```
*Expected Output: `ALL TESTS PASSED WITH 100% SUCCESS RATE!`*

### Step 4: Launch the Streamlit Application
```powershell
python -m streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

### Step 5: Run the Jupyter Notebook (.ipynb)
You can open and execute the complete 10-step academic workflow in Jupyter Notebook, VS Code, or Google Colab:
```powershell
python -m jupyter notebook dataset_preparation_eda.ipynb
```
*(Or simply open `dataset_preparation_eda.ipynb` directly in VS Code and click **Run All Cells**).*

---

## 8. College CIE 2 Unit I Evaluation Scheme (10 Steps to Demonstrate)

When presenting this project to your faculty, follow these 10 demonstration steps to score full marks:

| Step # | Evaluation Criterion | Module to Demonstrate | Key Action to Show Faculty |
| :---: | :--- | :--- | :--- |
| **Step 1** | **Dataset Ingestion** | Module 1 | Show default Titanic dataset loading (891 rows × 12 cols) and demonstrate custom CSV upload capability. |
| **Step 2** | **Structure & Schema** | Module 2 | Show `head()`, `tail()`, shape, and explain `dtypes` distinguishing numerical and categorical attributes. |
| **Step 3** | **Descriptive Summaries** | Module 2 | Navigate to Descriptive Statistics tab to show `describe()`, explaining mean, median, IQR, and skewness. |
| **Step 4** | **Missing Value Audit** | Module 3 | Show the missing summary table and bar chart exposing missingness in `Age` (177), `Cabin` (687), `Embarked` (2). |
| **Step 5** | **Imputation Treatment** | Module 3 | Apply **Median Imputation** for `Age`, **Mode Imputation** for `Embarked`, and **Drop Column** for `Cabin`. |
| **Step 6** | **Consistency & Duplicates** | Module 4 | Audit duplicates (0 found in Titanic) and demonstrate whitespace trimming and casing normalization tools. |
| **Step 7** | **Interactive EDA Visuals** | Module 5 | Demonstrate Histogram, Categorical Bar Chart, Box Plot (outliers), Scatter Plot, and Correlation Heatmap. |
| **Step 8** | **Categorical Encoding** | Module 6 | Apply One-Hot Encoding on `Sex` and `Embarked` with `drop_first=True` to prevent the dummy variable trap. |
| **Step 9** | **Feature Scaling** | Module 7 | Apply `StandardScaler` to `Age` and `Fare`, demonstrating the before vs. after distribution statistics ($\mu \approx 0, \sigma \approx 1$). |
| **Step 10**| **Verification & Export** | Modules 8 & 9 | Verify the zero-defect checklist, download `prepared_dataset.csv`, review the pipeline audit trail, and download the report. |

---

## 9. Results & Key Analytical Insights

1. **Age Distribution & Imputation:**  
   Passenger ages ranged from 0.42 to 80.0 years with positive skewness. Replacing 177 missing values with the median (28.0 years) maintained central tendency without being distorted by older passenger outliers.
2. **Cabin Column Pruning:**  
   `Cabin` possessed 77.10% missing values (687 of 891 records missing). Pruning this column was statistically justified because synthetic imputation across >70% nulls introduces spurious variance.
3. **Gender & Class Survival Disparity:**  
   Female passengers experienced a 74.2% survival rate compared to 18.9% for male passengers. First-class passengers achieved a 63.0% survival rate versus 24.2% for third-class passengers.
4. **Fare Scale Normalization:**  
   Raw ticket fares ranged from \$0.00 to \$512.33 ($\mu = \$32.20, \sigma = \$49.69$, skewness = +4.79). Applying `StandardScaler` brought `Fare` onto a standardized scale ($\mu = 0, \sigma = 1$), eliminating unit dominance over `Age`.

---

## 10. Limitations & Future Scope

### Limitations
- The current implementation handles tabular datasets (`.csv`). Unstructured data formats (images, audio, free text) are outside the scope of Unit I.
- Imputation methods currently include univariate strategies (mean, median, mode). Multivariate techniques (e.g. Iterative Imputer / KNN Imputer) require model fitting.

### Future Scope (Unit II & Beyond)
- Integration of automated train-test splitting (`train_test_split`) with strict data leakage prevention.
- Implementation of predictive supervised algorithms (Logistic Regression, Decision Trees, Random Forest).
- Model evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) in Unit III.

---

## 11. Academic Declaration

This project was developed by a Third-Year B.Tech Artificial Intelligence and Data Science student for the **Machine Learning (CIE 2: Unit I – Activity 1)** course. All implementations adhere strictly to academic guidelines, transparent data handling, and scientific preprocessing principles.

