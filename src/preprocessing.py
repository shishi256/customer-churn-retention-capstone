"""
Data Preprocessing and Feature Engineering Module.
Contains cleaning functions, advanced feature transformers,
and scikit-learn compatible ColumnTransformers.
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer churn dataframe:
    - Standardizes column names
    - Fixes string missing values in TotalCharges
    - Imputes zero-tenure customer total charges
    - Encodes binary target
    """
    data = df.copy()
    
    # 1. TotalCharges conversion
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # For zero tenure customers, total charges should be 0.0
        data["TotalCharges"] = data["TotalCharges"].fillna(0.0)
    
    # 2. Binary target encoding if present
    if "Churn" in data.columns:
        data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        
    return data

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-specific behavioral and risk features:
    - Service count aggregations
    - Financial burden and ratio metrics
    - High-risk interaction indicators
    - Support ticket velocity
    """
    data = df.copy()
    
    # Service counting
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    # Count how many addon services the customer has
    data["total_addon_services"] = 0
    for col in service_cols:
        if col in data.columns:
            data["total_addon_services"] += (data[col] == "Yes").astype(int)
            
    # Include phone service
    if "PhoneService" in data.columns:
        data["total_services"] = data["total_addon_services"] + (data["PhoneService"] == "Yes").astype(int)
    else:
        data["total_services"] = data["total_addon_services"]
        
    # Financial burden: charges per service
    data["avg_cost_per_service"] = data["MonthlyCharges"] / (data["total_services"] + 1)
    
    # Support ticket velocity (tickets per month of tenure)
    if "SupportTickets" in data.columns:
        data["ticket_velocity"] = data["SupportTickets"] / (data["tenure"] + 1)
        data["high_ticket_flag"] = (data["SupportTickets"] >= 3).astype(int)
    
    # High-risk cohort interaction: Month-to-month contract & Electronic Check
    if "Contract" in data.columns and "PaymentMethod" in data.columns:
        data["is_high_risk_contract_payment"] = (
            (data["Contract"] == "Month-to-month") & 
            (data["PaymentMethod"] == "Electronic check")
        ).astype(int)
        
    # Protective buffer: Customer has both TechSupport and OnlineSecurity
    if "TechSupport" in data.columns and "OnlineSecurity" in data.columns:
        data["has_security_and_support"] = (
            (data["TechSupport"] == "Yes") & (data["OnlineSecurity"] == "Yes")
        ).astype(int)
        
    # Engagement-to-ticket ratio
    if "LoginFrequency" in data.columns and "SupportTickets" in data.columns:
        data["engagement_friction_ratio"] = data["LoginFrequency"] / (data["SupportTickets"] + 1)
        
    # Tenure cohorts
    if "tenure" in data.columns:
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 60, 75],
            labels=["0-12m", "13-24m", "25-48m", "49-60m", "60m+"]
        ).astype(str)
        
    return data

def build_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """
    Constructs a robust scikit-learn ColumnTransformer.
    - Numerical: Median imputation + RobustScaler (resistant to ticket/charge outliers)
    - Categorical: Most frequent imputation + OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor

if __name__ == "__main__":
    raw_path = "data/raw/customer_churn_raw.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        df_clean = clean_data(df_raw)
        df_feat = engineer_features(df_clean)
        os.makedirs("data/processed", exist_ok=True)
        processed_path = "data/processed/customer_churn_processed.csv"
        df_feat.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path}")
        print(f"Processed shape: {df_feat.shape}")
        print(f"New engineered columns: {[c for c in df_feat.columns if c not in df_raw.columns]}")

