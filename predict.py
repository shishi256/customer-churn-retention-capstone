"""
Customer Churn Risk Scoring & Decision Support CLI Tool.
Loads the trained production pipeline and generates real-time customer risk scores,
actionable drivers, and recommended retention interventions.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import engineer_features

def load_pipeline():
    model = joblib.load("models/histgradientboosting.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    return model, preprocessor

def score_customer(customer_data: dict, optimal_threshold: float = 0.35) -> dict:
    """
    Scores an individual customer dictionary and returns:
    - Churn probability
    - Risk tier (Low, Medium, Critical)
    - Decision action
    - Recommended proactive intervention
    """
    model, preprocessor = load_pipeline()
    
    df = pd.DataFrame([customer_data])
    df_engineered = engineer_features(df)
    
    # Preprocess
    X_trans = preprocessor.transform(df_engineered)
    prob = float(model.predict_proba(X_trans)[0, 1])
    
    # Risk Classification
    if prob >= 0.65:
        tier = "Critical Risk"
        action = "Priority Concierge Outreach + Annual Contract Discount Offer (20% off)"
    elif prob >= optimal_threshold:
        tier = "Elevated Risk"
        action = "Automated Retention Campaign: Free 3-Month TechSupport & Cybersecurity Upgrade"
    elif prob >= 0.20:
        tier = "Moderate / Watchlist"
        action = "Digital Nudge: Highlight Mobile App Benefits & Auto-Pay $5/mo Discount"
    else:
        tier = "Low Risk (Stable Loyal)"
        action = "Standard Lifecycle Engagement / Referral Program Offer"
        
    return {
        "customerID": customer_data.get("customerID", "Unknown"),
        "churn_probability": round(prob, 4),
        "predicted_churn": bool(prob >= optimal_threshold),
        "risk_tier": tier,
        "recommended_action": action
    }

if __name__ == "__main__":
    sample_customer = {
        "customerID": "DEMO-9999",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 268.50,
        "SupportTickets": 4,
        "LoginFrequency": 11,
        "SatisfactionScore": 2
    }
    
    print("Evaluating sample high-risk customer profile...")
    result = score_customer(sample_customer)
    print(json.dumps(result, indent=2))

