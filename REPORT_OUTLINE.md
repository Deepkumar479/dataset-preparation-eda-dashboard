# ACADEMIC PROJECT REPORT OUTLINE
**COURSE:** Machine Learning | **SEMESTER:** VI (Third-Year B.Tech)  
**PROGRAM:** Artificial Intelligence and Data Science  
**ACTIVITY:** Continuous Internal Evaluation (CIE 2) – Unit I Activity 1  
**PROJECT TITLE:** Dataset Preparation and Exploratory Data Analysis (EDA) Dashboard  

---

## FRONT MATTER (Preliminary Pages)

### 1. Title Page
- **Institution Name:** Department of Artificial Intelligence and Data Science, [College/University Name]
- **Project Title:** *Dataset Preparation and Exploratory Data Analysis (EDA) Dashboard: An Interactive System for Data Inspection, Cleaning, Analysis, and Preprocessing for Machine Learning*
- **Submission Statement:** *A Project Report submitted in partial fulfillment of the requirements for Continuous Internal Evaluation 2 (CIE 2) in Machine Learning.*
- **Submitted By:** [Student Full Name] (PRN / Roll Number: [Your Roll No])
- **Under the Guidance of:** [Faculty Guide Name, Designation]
- **Academic Year:** 2026–2027

### 2. Certificate of Originality
- Official departmental certificate signed by Project Guide, Head of Department (HOD), and External/Internal Examiner certifying that the work was carried out by the student.

### 3. Declaration
- Student formal statement confirming that the project represents original work and adheres to academic integrity standards.

### 4. Acknowledgements
- Formal gratitude to Faculty Guide, Head of Department, Laboratory Assistants, and Peers.

### 5. Abstract
- A concise 250–300 word executive summary highlighting:
  - The critical role of data conditioning prior to mathematical machine learning modeling.
  - The development of an interactive Streamlit dashboard covering 9 specialized modules.
  - Key statistical findings on the Titanic dataset (e.g., median age imputation, cabin column pruning, fare standardisation).
  - The generation of a zero-defect prepared dataset (`prepared_dataset.csv`).

### 6. Table of Contents, List of Figures, List of Tables

---

## CHAPTER-WISE REPORT BODY

### CHAPTER 1: INTRODUCTION
- **1.1 Background & Context:**
  - The Machine Learning Lifecycle: CRISP-DM framework and the 80/20 data preparation principle.
  - Significance of Unit I: Understanding data representations before algorithm selection.
- **1.2 Problem Statement:**
  - Inherent defects in raw real-world data (missing values, disparate feature scales, string representations).
  - The need for an interactive graphical system combining educational theory with automated preprocessing.
- **1.3 Project Objectives:**
  - Automated inspection using fundamental Pandas methods (`head`, `tail`, `info`, `describe`).
  - Statistically justified missing-value imputation (Mean, Median, Mode, Pruning).
  - Elimination of duplicate records and whitespace inconsistencies.
  - Comprehensive exploratory data analysis across 5 visualization paradigms.
  - Multi-attribute categorical encoding (One-Hot Encoding with `drop_first=True`).
  - Numerical scaling via StandardScaler and MinMaxScaler.
  - Generation of an exportable, ML-ready dataset without altering raw memory state.
- **1.4 Scope & Boundaries:**
  - Strict compliance with Unit I: Focus on data engineering; predictive models (Regression, Trees) are strictly out of scope.

### CHAPTER 2: THEORETICAL FOUNDATIONS & LITERATURE REVIEW
- **2.1 The Principle of "Garbage In, Garbage Out" (GIGO) in Machine Learning.**
- **2.2 Missing Data Mechanisms:**
  - Missing Completely at Random (MCAR).
  - Missing at Random (MAR).
  - Missing Not at Random (MNAR).
  - Comparative analysis: Mean Imputation vs. Median Imputation vs. Structural Pruning.
- **2.3 Redundancy and Sample Bias:**
  - Impact of duplicate instances on empirical risk minimization (loss weight inflation).
- **2.4 Mathematical Formulations of Feature Preprocessing:**
  - *One-Hot Encoding:* Orthonormal basis vector representation and the Dummy Variable Trap (Multicollinearity).
  - *StandardScaler (Z-Score Standardisation):* $z = \frac{x - \mu}{\sigma}$ (Centers mean at 0, standard deviation at 1).
  - *MinMaxScaler (Range Normalisation):* $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ (Bounds values in $[0, 1]$).
- **2.5 Data Leakage Prevention:**
  - The theoretical requirement of fitting scalers on training distributions only.

### CHAPTER 3: SYSTEM ARCHITECTURE & METHODOLOGY
- **3.1 High-Level Architecture Diagram:**
  - Client Browser (Streamlit UI) $\leftrightarrow$ Python Runtime Engine $\leftrightarrow$ Modular Processing Components $\leftrightarrow$ In-Memory State Manager (`st.session_state`).
- **3.2 Directory and Code Organization:**
  - Explanation of `app.py`, `assets/style.css`, and the 7 specialized modules in `modules/`.
- **3.3 State Management & Memory Safety:**
  - Independent tracking of `raw_df` (immutable) and `working_df` (active transformations).
  - Chronological pipeline audit logging (`audit_log`).
- **3.4 Technologies & Libraries Utilized:**
  - Streamlit, Pandas, NumPy, Plotly, Scikit-learn, Seaborn, Matplotlib.

### CHAPTER 4: DATA QUALITY AUDIT & CLEANING
- **4.1 Dataset Profile & Ingestion:**
  - Demographics of the Kaggle Titanic demonstration dataset (891 rows, 12 features).
- **4.2 Structural Inspection Results:**
  - Memory footprint analysis and data type inference table.
- **4.3 Missing Value Audit & Imputation Engineering:**
  - Identification of nulls in `Age` (177 cells, 19.87%), `Cabin` (687 cells, 77.10%), and `Embarked` (2 cells, 0.22%).
  - Statistical justification for Median Imputation on `Age` (resistance to skewness).
  - Statistical justification for Mode Imputation on `Embarked` (discrete nominal mode 'S').
  - Statistical justification for Column Deletion on `Cabin` (>70% null threshold).
- **4.4 Data Consistency & Deduplication:**
  - Exact duplicate scan results.
  - Detection and automated stripping of leading/trailing whitespace in string features.
  - Categorical casing standardization.

### CHAPTER 5: EXPLORATORY DATA ANALYSIS (EDA) & STATISTICAL INSIGHTS
- **5.1 Univariate Numerical Analysis:**
  - Descriptive statistics (Mean, Median, Std Dev, Min, Max, Quartiles, Skewness, IQR).
  - Distribution histograms and kernel density visualizations.
- **5.2 Univariate Categorical Analysis:**
  - Value counts and frequency distributions for `Pclass`, `Sex`, and `Embarked`.
- **5.3 Outlier Detection via Tukey's Method:**
  - Box plot visual analysis; identification of extreme high-value outliers in `Fare`.
- **5.4 Bivariate Interactions:**
  - Scatter plot inspection of `Age` vs. `Fare` segmented by class and survival.
- **5.5 Pearson Feature Correlation Analysis:**
  - Correlation heatmap evaluation; diagnosing inter-feature linear dependencies.
- **5.6 Key Authentic Empirical Insights:**
  - Gender survival disparity (Female 74.2% vs. Male 18.9%).
  - Class-based survival hierarchy (1st class 63.0% vs. 3rd class 24.2%).
  - Monetary right-skewness of `Fare` (+4.79 skewness).

### CHAPTER 6: FEATURE PREPROCESSING (ENCODING & SCALING)
- **6.1 Categorical Encoding Implementation:**
  - One-Hot Encoding on nominal features `Sex` and `Embarked` with `drop_first=True`.
  - Exclusion of high-cardinality identity keys (`PassengerId`, `Name`, `Ticket`).
  - Resulting dimensional expansion and dummy variable verification.
- **6.2 Numerical Feature Scaling Implementation:**
  - Application of `StandardScaler` to continuous features `Age` and `Fare`.
  - Statistical verification of post-scaling distributions ($\mu \approx 0.000, \sigma \approx 1.000$).
  - Graphical distribution comparison (Before vs. After).

### CHAPTER 7: RESULTS & PREPARED DATASET VERIFICATION
- **7.1 Machine Learning Readiness Audit:**
  - Verification of 0 missing values across all records.
  - Verification of 0 duplicate records.
  - Verification of 100% numerical data type compatibility.
- **7.2 Tabular Comparison: Raw vs. Prepared Dataset:**
  - Side-by-side metric matrix (Rows, Columns, Nulls, Duplicates, Memory Size).
- **7.3 Prepared Dataset Export:**
  - Inspection of `outputs/prepared_dataset.csv`.
- **7.4 Session Pipeline Audit Trail Review:**
  - Step-by-step audit log output.

### CHAPTER 8: CONCLUSION & FUTURE SCOPE
- **8.1 Summary of Unit I Achievements:**
  - Complete compliance with CIE 2 Activity 1 guidelines.
- **8.2 Practical Takeaways:**
  - Why intuitive visual inspection enhances data science engineering.
- **8.3 Future Scope for Unit II (Supervised Learning):**
  - Partitioning into train/test sets using stratified sampling.
  - Fitting Logistic Regression, Decision Trees, and Random Forests on `prepared_dataset.csv`.
  - Hyperparameter tuning and model performance evaluation in Unit III.

---

## BACK MATTER

### References (IEEE Format)
1. J. Han, M. Kamber, and J. Pei, *Data Mining: Concepts and Techniques*, 3rd ed. Morgan Kaufmann, 2011.
2. S. Raschka and V. Mirjalili, *Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow*, 3rd ed. Packt Publishing, 2019.
3. F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
4. W. McKinney, *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter*, 3rd ed. O'Reilly Media, 2022.
5. Kaggle, "Titanic: Machine Learning from Disaster," [Online]. Available: https://www.kaggle.com/datasets/yasserh/titanic-dataset.

---

## APPENDIX: FACULTY VIVA-VOCE EXAMINATION CHEAT SHEET

| Question | Recommended Technical Answer |
| :--- | :--- |
| **Q1: Why did you choose Median instead of Mean to impute missing values in `Age`?** | "The `Age` distribution has positive skewness with several older passenger outliers. The arithmetic mean is sensitive to extreme values, which would artificially pull imputed ages upward. The median is a non-parametric statistic representing the true central 50th percentile, making it robust against outlier distortion." |
| **Q2: Why was `Cabin` dropped instead of imputed?** | "`Cabin` contains 687 missing records out of 891 (77.10% missingness). In machine learning data preparation, when a feature exceeds a 50%–70% missingness threshold, synthetic imputation creates an overwhelming amount of artificial noise and pseudo-data that deceives model optimization. Dropping it preserves structural signal." |
| **Q3: What is the Dummy Variable Trap, and how did your dashboard prevent it?** | "The dummy variable trap occurs when all $k$ nominal categories are represented with $k$ binary columns in the presence of an intercept term. Because the sum of all dummy indicators equals 1, the columns exhibit perfect multicollinearity ($X^T X$ is singular and non-invertible). By setting `drop_first=True`, we encode $k$ categories using $k-1$ binary columns, completely eliminating linear dependence." |
| **Q4: Why must Feature Scalers be fit only on the training set in production ML?** | "Fitting a scaler across an entire combined dataset calculates global statistics ($\mu, \sigma, min, max$) that incorporate knowledge from the test/validation set. This constitutes Data Leakage, which yields overly optimistic validation metrics that collapse when deployed on unseen real-world data." |
| **Q5: Why did you not encode `Name` or `Ticket`?** | "`Name` and `Ticket` are high-cardinality unique identifiers possessing unique or near-unique string values per passenger. One-hot encoding them would generate hundreds of sparse binary columns, triggering the Curse of Dimensionality and severe overfitting without generalizable predictive signal." |

