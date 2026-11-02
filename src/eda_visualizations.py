"""
Exploratory Data Analysis (EDA) and Visualization Module.
Generates comprehensive publication-quality figures illustrating
churn drivers, customer distributions, and feature relationships.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = ["#1f77b4", "#ff7f0e"]

def generate_eda_figures(data_path="data/processed/customer_churn_processed.csv", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df["Churn_Label"] = df["Churn"].map({1: "Churned", 0: "Retained"})
    
    # -------------------------------------------------------------
    # Figure 1: Churn Target Distribution and Tenure Dynamics
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1A: Target Class Distribution Pie & Bar
    counts = df["Churn_Label"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#2ca02c", "#d62728"], alpha=0.85, width=0.5)
    for i, v in enumerate(counts.values):
        pct = (v / len(df)) * 100
        axes[0].text(i, v + 100, f"{v:,} ({pct:.1f}%)", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_title("Customer Attrition vs Retention (Class Imbalance)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Customer Count", fontsize=11)
    axes[0].set_ylim(0, max(counts.values) * 1.15)
    
    # 1B: Tenure Distribution by Churn
    sns.histplot(
        data=df, x="tenure", hue="Churn_Label", kde=True,
        element="step", common_norm=False, palette=["#2ca02c", "#d62728"], ax=axes[1]
    )
    axes[1].set_title("Tenure Distribution by Churn Status (Months)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Tenure (Months)", fontsize=11)
    axes[1].set_ylabel("Density / Count", fontsize=11)
    axes[1].axvline(df[df['Churn']==1]['tenure'].median(), color="#d62728", linestyle="--", label=f"Churn Median: {df[df['Churn']==1]['tenure'].median():.0f} mo")
    axes[1].axvline(df[df['Churn']==0]['tenure'].median(), color="#2ca02c", linestyle="--", label=f"Retained Median: {df[df['Churn']==0]['tenure'].median():.0f} mo")
    axes[1].legend()
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "01_churn_distribution_and_tenure.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Generated: {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Contract Type and Payment Method Churn Risk
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2A: Churn Rate by Contract Type
    contract_churn = df.groupby("Contract")["Churn"].agg(["count", "mean"]).reset_index()
    contract_churn["mean"] = contract_churn["mean"] * 100
    sns.barplot(data=contract_churn, x="Contract", y="mean", palette="Blues_r", ax=axes[0])
    for i, row in contract_churn.iterrows():
        axes[0].text(i, row["mean"] + 1.0, f"{row['mean']:.1f}% (N={row['count']})", ha="center", fontweight="bold")
    axes[0].set_title("Churn Rate by Contract Horizon", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)", fontsize=11)
    axes[0].set_ylim(0, max(contract_churn["mean"]) * 1.2)
    
    # 2B: Churn Rate by Payment Method
    pay_churn = df.groupby("PaymentMethod")["Churn"].agg(["count", "mean"]).reset_index()
    pay_churn["mean"] = pay_churn["mean"] * 100
    pay_churn = pay_churn.sort_values(by="mean", ascending=False)
    sns.barplot(data=pay_churn, x="mean", y="PaymentMethod", palette="Reds_r", ax=axes[1])
    for i, row in pay_churn.reset_index(drop=True).iterrows():
        axes[1].text(row["mean"] + 0.5, i, f" {row['mean']:.1f}%", va="center", fontweight="bold")
    axes[1].set_title("Churn Rate by Payment Method", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Churn Rate (%)", fontsize=11)
    axes[1].set_xlim(0, max(pay_churn["mean"]) * 1.2)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "02_contract_and_billing_impact.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Generated: {fig2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Support Friction & Customer Satisfaction
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # 3A: Churn Rate by Support Ticket Count
    ticket_churn = df.groupby("SupportTickets")["Churn"].agg(["count", "mean"]).reset_index()
    ticket_churn["mean"] = ticket_churn["mean"] * 100
    sns.barplot(data=ticket_churn, x="SupportTickets", y="mean", palette="Oranges", ax=axes[0])
    for i, row in ticket_churn.iterrows():
        axes[0].text(i, row["mean"] + 1.2, f"{row['mean']:.0f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Churn Escalation Across Support Ticket Volumes", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Support Tickets Filed", fontsize=11)
    axes[0].set_ylabel("Churn Rate (%)", fontsize=11)
    axes[0].set_ylim(0, 105)
    
    # 3B: Satisfaction Score Distribution
    sns.countplot(data=df, x="SatisfactionScore", hue="Churn_Label", palette=["#2ca02c", "#d62728"], ax=axes[1])
    axes[1].set_title("Customer Volume by Satisfaction Score (1-5)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Satisfaction Score (1 = Poor, 5 = Excellent)", fontsize=11)
    axes[1].set_ylabel("Customer Count", fontsize=11)
    axes[1].legend(title="Status")
    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "03_support_and_satisfaction_friction.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Generated: {fig3_path}")
    
    # -------------------------------------------------------------
    # Figure 4: Financial Load (Monthly Charges vs Total Services)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # 4A: Monthly Charges Distribution
    sns.boxplot(data=df, x="Churn_Label", y="MonthlyCharges", palette=["#2ca02c", "#d62728"], ax=axes[0], width=0.4)
    axes[0].set_title("Monthly Recurring Charges Distribution by Churn", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Monthly Charges ($)", fontsize=11)
    axes[0].set_xlabel("Customer Status", fontsize=11)
    
    # 4B: Total Services Subscribed vs Churn Rate
    svc_churn = df.groupby("total_services")["Churn"].agg(["count", "mean"]).reset_index()
    svc_churn["mean"] = svc_churn["mean"] * 100
    sns.lineplot(data=svc_churn, x="total_services", y="mean", marker="o", color="#d62728", linewidth=2.5, markersize=8, ax=axes[1])
    for _, row in svc_churn.iterrows():
        axes[1].text(row["total_services"], row["mean"] + 1.5, f"{row['mean']:.1f}%", ha="center", fontweight="bold")
    axes[1].set_title("Churn Rate vs Total Products & Services Subscribed", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Subscribed Services (Ecosystem Stickiness)", fontsize=11)
    axes[1].set_ylabel("Churn Rate (%)", fontsize=11)
    axes[1].set_ylim(0, max(svc_churn["mean"]) * 1.25)
    
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "04_financial_load_and_services.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Generated: {fig4_path}")
    
    # -------------------------------------------------------------
    # Figure 5: Numerical Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 8.5))
    num_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "SupportTickets",
        "LoginFrequency", "SatisfactionScore", "total_services",
        "avg_cost_per_service", "ticket_velocity", "Churn"
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-0.8, vmax=0.8, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f"
    )
    plt.title("Correlation Matrix of Numerical and Engineered Metrics", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "05_numerical_correlation_matrix.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Generated: {fig5_path}")

if __name__ == "__main__":
    generate_eda_figures()

