"""
Error Analysis, Calibration, and Cost-Benefit Decision Optimization Module.
Analyzes model failure modes (False Positives and False Negatives),
evaluates probability calibration curves, and optimizes decision thresholds
using asymmetric financial business payoff matrices.
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
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, brier_score_loss, classification_report

def run_error_and_threshold_analysis(data_path="data/processed/customer_churn_processed.csv"):
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # Load best model and preprocessor
    # We will check if HistGradientBoosting or Random Forest exists, else fallback to best available
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
    y_prob = model.predict_proba(X_test_trans)[:, 1]
    
    # -------------------------------------------------------------
    # 1. Asymmetric Financial Cost-Benefit Decision Optimization
    # -------------------------------------------------------------
    # Business Parameters:
    # Average Customer Lifetime Value saved (annualized): $720
    # Cost of retention outreach campaign (offer + agent contact): $65
    # Retention intervention success rate (uplift/save rate): 50%
    # Value of True Positive: SaveRate * LTV - Cost = 0.50 * 720 - 65 = $295
    # Value of False Positive: -$65 (wasted promotion on non-churner)
    # Value of False Negative: $0 net direct cash outlay, but -$360 in missed revenue opportunity
    # Value of True Negative: $0
    
    LTV = 720.0
    PROMO_COST = 65.0
    SAVE_RATE = 0.50
    
    thresholds = np.linspace(0.05, 0.95, 91)
    profits = []
    recalls = []
    precisions = []
    
    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_t).ravel()
        
        # Total Net Value Generated on Test Set
        # Each TP saves 0.5 * 720 - 65 = 295
        # Each FP wastes 65
        net_profit = (tp * (SAVE_RATE * LTV - PROMO_COST)) - (fp * PROMO_COST)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        profits.append(net_profit)
        recalls.append(recall)
        precisions.append(precision)
        
    best_idx = np.argmax(profits)
    best_threshold = thresholds[best_idx]
    max_profit = profits[best_idx]
    
    # Compare with standard 0.50 threshold
    idx_50 = np.argmin(np.abs(thresholds - 0.50))
    profit_50 = profits[idx_50]
    profit_lift = max_profit - profit_50
    
    print(f"Optimal Decision Threshold: {best_threshold:.2f}")
    print(f"Projected Net Profit at Optimal: ${max_profit:,.2f}")
    print(f"Projected Net Profit at 0.50: ${profit_50:,.2f}")
    print(f"Net Profit Lift from Optimization: ${profit_lift:,.2f} (+{(profit_lift / profit_50) * 100:.1f}%)")
    
    # Save threshold results
    thresh_df = pd.DataFrame({
        "Threshold": thresholds,
        "Net_Profit": profits,
        "Recall": recalls,
        "Precision": precisions
    })
    thresh_df.to_csv("reports/threshold_optimization.csv", index=False)
    
    # -------------------------------------------------------------
    # Figure 9: Profit Curve vs Decision Threshold
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = "#1f77b4"
    ax1.set_xlabel("Classification Probability Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Net Campaign Profit ($ USD)", color=color, fontsize=11, fontweight="bold")
    line1 = ax1.plot(thresholds, profits, color=color, linewidth=2.5, label="Net Profit ($)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axvline(best_threshold, color="#d62728", linestyle="--", linewidth=1.8, label=f"Optimal Threshold ({best_threshold:.2f})")
    ax1.axvline(0.50, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Default 0.50 Threshold")
    ax1.scatter([best_threshold], [max_profit], color="#d62728", s=100, zorder=5)
    ax1.annotate(
        f"Max Profit: ${max_profit:,.0f}\n@ Threshold {best_threshold:.2f}",
        xy=(best_threshold, max_profit),
        xytext=(best_threshold + 0.08, max_profit - 10000),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold"
    )
    
    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("Metric Rate", color=color2, fontsize=11, fontweight="bold")
    line2 = ax2.plot(thresholds, recalls, color="#2ca02c", linestyle="-.", label="Recall")
    line3 = ax2.plot(thresholds, precisions, color="#ff7f0e", linestyle="--", label="Precision")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.05)
    
    lines = line1 + line2 + line3 + [ax1.get_lines()[1], ax1.get_lines()[2]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center", framealpha=0.9)
    plt.title("Business Value Optimization: Financial Profit vs Decision Threshold", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig9_path = "figures/09_expected_profit_vs_threshold.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"Generated: {fig9_path}")
    
    # -------------------------------------------------------------
    # Figure 10: Probability Calibration Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#1f77b4", label="Model Probability Calibration")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    plt.title(f"Reliability Diagram (Calibration Curve)\nBrier Score: {brier_score_loss(y_test, y_prob):.4f}", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Churners", fontsize=11)
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig10_path = "figures/10_calibration_curves.png"
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"Generated: {fig10_path}")
    
    # -------------------------------------------------------------
    # Detailed Error Decomposition (FP vs FN)
    # -------------------------------------------------------------
    optimal_pred = (y_prob >= best_threshold).astype(int)
    test_eval_df = X_test.copy()
    test_eval_df["True_Churn"] = y_test.values
    test_eval_df["Predicted_Prob"] = np.round(y_prob, 4)
    test_eval_df["Predicted_Churn"] = optimal_pred
    
    test_eval_df["Error_Type"] = "True Negative (Correct Retained)"
    test_eval_df.loc[(test_eval_df["True_Churn"] == 1) & (test_eval_df["Predicted_Churn"] == 1), "Error_Type"] = "True Positive (Correct Churn)"
    test_eval_df.loc[(test_eval_df["True_Churn"] == 0) & (test_eval_df["Predicted_Churn"] == 1), "Error_Type"] = "False Positive (False Alarm)"
    test_eval_df.loc[(test_eval_df["True_Churn"] == 1) & (test_eval_df["Predicted_Churn"] == 0), "Error_Type"] = "False Negative (Missed Churn)"
    
    error_summary = test_eval_df.groupby("Error_Type")[["tenure", "MonthlyCharges", "SupportTickets", "SatisfactionScore"]].mean()
    error_counts = test_eval_df["Error_Type"].value_counts().rename("Customer_Count")
    error_profile = pd.concat([error_counts, error_summary], axis=1)
    error_profile.to_csv("reports/error_analysis_profiles.csv")
    print("\n=== ERROR ANALYSIS PROFILES ===")
    print(error_profile.to_string())

if __name__ == "__main__":
    run_error_and_threshold_analysis()

