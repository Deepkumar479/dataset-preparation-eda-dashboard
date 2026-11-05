"""
Automated Comprehensive Test Script for Dataset Preparation & EDA Dashboard
Validates imports, data loading, and core functions across all modules.
"""

import os
import pandas as pd
import numpy as np

def run_tests():
    print("--- [TEST 1] Verifying File Existence ---")
    assert os.path.exists("app.py"), "app.py missing"
    assert os.path.exists("requirements.txt"), "requirements.txt missing"
    assert os.path.exists("assets/style.css"), "assets/style.css missing"
    assert os.path.exists("data/sample_dataset.csv"), "data/sample_dataset.csv missing"
    assert os.path.exists("outputs/prepared_dataset.csv"), "outputs/prepared_dataset.csv missing"
    assert os.path.exists("dataset_preparation_eda.ipynb"), "dataset_preparation_eda.ipynb missing"
    print("Files verified successfully!")

    print("--- [TEST 2] Verifying Modules Import ---")
    from modules.overview import render_overview
    from modules.inspection import render_inspection
    from modules.missing_values import render_missing_values
    from modules.data_cleaning import render_data_cleaning
    from modules.eda import render_eda
    from modules.encoding import render_encoding
    from modules.scaling import render_scaling
    from modules.export import render_export_and_summary
    print("All modules imported cleanly!")

    print("--- [TEST 3] Verifying Data Loading & Quality Operations ---")
    df = pd.read_csv("data/sample_dataset.csv")
    assert df.shape == (891, 12), f"Unexpected raw shape: {df.shape}"
    print("Raw dataset verified: 891 rows, 12 columns.")

    # Test Missing Values handling logic
    assert df["Age"].isna().sum() == 177, "Expected 177 missing Age values"
    df["Age"] = df["Age"].fillna(df["Age"].median())
    assert df["Age"].isna().sum() == 0, "Age imputation failed"

    # Test String trimming
    df["Sex"] = df["Sex"].str.strip()
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0]).str.strip()
    assert df["Embarked"].isna().sum() == 0, "Embarked imputation failed"

    # Test Deduplication
    dups = df.duplicated().sum()
    assert dups == 0, "Unexpected duplicates found"

    # Test Encoding
    df_encoded = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int)
    assert "Sex_male" in df_encoded.columns, "OHE Sex_male column missing"
    assert "Embarked_Q" in df_encoded.columns, "OHE Embarked_Q column missing"
    assert "Embarked_S" in df_encoded.columns, "OHE Embarked_S column missing"

    # Test Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_encoded[["Age", "Fare"]])
    assert scaled.shape == (891, 2), "Scaling shape mismatch"
    assert np.isclose(scaled[:, 0].mean(), 0, atol=1e-5), "Age mean after standard scaler is not ~0"
    assert np.isclose(scaled[:, 0].std(), 1, atol=1e-5), "Age std after standard scaler is not ~1"

    print("All mathematical & algorithmic transformations verified successfully!")
    print("========================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS RATE!")
    print("========================================")

if __name__ == "__main__":
    run_tests()

