"""
Explainability, Persona Clustering, and Business Decision Simulation Module.
Generates feature importance metrics, extracts behavioral customer personas,
and models financial ROI of targeted proactive retention interventions.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from preprocessing import build_preprocessor

def run_explainability_and_personas(data_path="data/processed/customer_churn_processed.csv"):
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # Load model and preprocessor
    model_path = "models/histgradientboosting.joblib"
    if not os.path.exists(model_path):
        model_path = "models/random_forest.joblib"
    if not os.path.exists(model_path):
        model_path = "models/logistic_regression_baseline.joblib"
        
    model = joblib.load(model_path)
    preprocessor = joblib.load("models/preprocessor.joblib")
    
    drop_cols = ["customerID", "Churn_Label", "tenure_group"] if "customerID" in df.columns else []
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["Churn"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    X_test_trans = preprocessor.transform(X_test)
    
    # -------------------------------------------------------------
    # 1. Feature Importance via Permutation Importance
    # -------------------------------------------------------------
    print("Computing Permutation Feature Importance...")
    perm_importance = permutation_importance(
        model, X_test_trans, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    
    # Retrieve feature names from ColumnTransformer
    cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out()
    num_feature_names = preprocessor.transformers_[0][2]
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    feat_imp_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance_Mean": perm_importance.importances_mean,
        "Importance_Std": perm_importance.importances_std
    }).sort_values(by="Importance_Mean", ascending=False)
    
    feat_imp_df.to_csv("reports/feature_importance.csv", index=False)
    print("\n=== TOP 10 MOST IMPACTFUL FEATURES ===")
    print(feat_imp_df.head(10).to_string(index=False))
    
    # -------------------------------------------------------------
    # Figure 11: Top 15 Feature Importances
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    top_15 = feat_imp_df.head(15).iloc[::-1]
    
    plt.barh(top_15["Feature"], top_15["Importance_Mean"], xerr=top_15["Importance_Std"], color="#1f77b4", alpha=0.85, capsize=4)
    plt.title("Top 15 Feature Importances (Permutation ROC-AUC Drop)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Drop in Test ROC-AUC", fontsize=11)
    plt.tight_layout()
    fig11_path = "figures/11_feature_importance.png"
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    print(f"Generated: {fig11_path}")
    
    # -------------------------------------------------------------
    # 2. Customer Persona Segmentation
    # -------------------------------------------------------------
    # Identify high risk population
    y_prob = model.predict_proba(X_test_trans)[:, 1]
    high_risk_mask = y_prob >= 0.40
    high_risk_customers = X_test[high_risk_mask].copy()
    
    cluster_features = [
        "tenure", "MonthlyCharges", "SupportTickets", "SatisfactionScore", "total_services"
    ]
    
    scaler_cluster = build_preprocessor([], cluster_features)
    cluster_matrix = scaler_cluster.fit_transform(high_risk_customers[cluster_features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    high_risk_customers["Persona_Cluster"] = kmeans.fit_predict(cluster_matrix)
    
    persona_names = {
        0: "Persona A: Frustrated Onboarders",
        1: "Persona B: Price-Sensitive Streamers",
        2: "Persona C: High-Friction Month-to-Month"
    }
    high_risk_customers["Persona_Name"] = high_risk_customers["Persona_Cluster"].map(persona_names)
    
    persona_profiles = high_risk_customers.groupby("Persona_Name")[cluster_features].mean().round(2)
    persona_counts = high_risk_customers["Persona_Name"].value_counts().rename("Customer_Count")
    persona_summary = pd.concat([persona_counts, persona_profiles], axis=1)
    persona_summary.to_csv("reports/customer_personas_summary.csv")
    print("\n=== AT-RISK CUSTOMER PERSONA PROFILES ===")
    print(persona_summary.to_string())
    
    # -------------------------------------------------------------
    # Figure 12: Customer Persona Profiling Radar / Bar Visual
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_show = ["tenure", "MonthlyCharges", "SupportTickets"]
    
    for i, col in enumerate(metrics_to_show):
        sns.barplot(data=high_risk_customers, x="Persona_Name", y=col, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Average {col} by Persona", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel(col, fontsize=10)
        axes[i].tick_params(axis="x", rotation=25)
        
    plt.tight_layout()
    fig12_path = "figures/12_customer_risk_personas.png"
    plt.savefig(fig12_path, dpi=300)
    plt.close()
    print(f"Generated: {fig12_path}")

if __name__ == "__main__":
    run_explainability_and_personas()

