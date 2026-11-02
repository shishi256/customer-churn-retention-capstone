"""
Machine Learning Training, Benchmarking, and Evaluation Module.
Trains and compares multiple classifier architectures with 5-fold Stratified CV,
class imbalance handling, metric evaluations (ROC-AUC, PR-AUC, F1, F2, Brier Score),
and exports trained pipelines and visualization artifacts.
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)

from preprocessing import build_preprocessor

def prepare_data(data_path="data/processed/customer_churn_processed.csv"):
    df = pd.read_csv(data_path)
    
    # Drop identifiers and non-predictive meta
    drop_cols = ["customerID", "Churn_Label", "tenure_group"] if "customerID" in df.columns else []
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["Churn"])
    y = df["Churn"]
    
    # Separate categorical and numerical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    return X, y, cat_cols, num_cols

def train_and_evaluate(data_path="data/processed/customer_churn_processed.csv"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    X, y, cat_cols, num_cols = prepare_data(data_path)
    
    # Train-test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Save fitted preprocessor
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    
    # Define Candidate Models
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_depth=6,
            class_weight="balanced", random_state=42
        ),
        "Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=400,
            alpha=0.01, random_state=42, early_stopping=True
        )
    }
    
    benchmark_records = []
    trained_models = {}
    test_predictions = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in models.items():
        print(f"\n--- Evaluating: {name} ---")
        
        # 5-Fold Stratified Cross Validation on Training Set
        cv_scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_res = cross_validate(clf, X_train_trans, y_train, cv=cv, scoring=cv_scoring, n_jobs=-1)
        
        # Fit on full training set
        clf.fit(X_train_trans, y_train)
        trained_models[name] = clf
        joblib.dump(clf, f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
        
        # Evaluate on Holdout Test Set
        y_pred = clf.predict(X_test_trans)
        y_prob = clf.predict_proba(X_test_trans)[:, 1] if hasattr(clf, "predict_proba") else y_pred
        test_predictions[name] = {"pred": y_pred, "prob": y_prob}
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        print(f"Test ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | F2: {f2:.4f} | Recall: {rec:.4f} | Brier: {brier:.4f}")
        
        benchmark_records.append({
            "Model": name,
            "CV_ROC_AUC_Mean": cv_res["test_roc_auc"].mean(),
            "CV_ROC_AUC_Std": cv_res["test_roc_auc"].std(),
            "CV_F1_Mean": cv_res["test_f1"].mean(),
            "Test_Accuracy": acc,
            "Test_Precision": prec,
            "Test_Recall": rec,
            "Test_F1": f1,
            "Test_F2_Score": f2,
            "Test_ROC_AUC": roc_auc,
            "Test_PR_AUC": pr_auc,
            "Test_Brier_Score": brier
        })
        
    benchmark_df = pd.DataFrame(benchmark_records).sort_values(by="Test_ROC_AUC", ascending=False)
    benchmark_df.to_csv("reports/model_benchmark_results.csv", index=False)
    print("\n=== MODEL BENCHMARK RESULTS ===")
    print(benchmark_df.to_string(index=False))
    
    # -------------------------------------------------------------
    # Figure 6: Model Comparison Bar Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ["Test_ROC_AUC", "Test_PR_AUC", "Test_F1", "Test_F2_Score"]
    plot_df = benchmark_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    plot_df["Metric"] = plot_df["Metric"].str.replace("Test_", "").str.replace("_", "-")
    
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="tab10")
    plt.title("Performance Comparison Across Machine Learning Architectures", fontsize=14, fontweight="bold")
    plt.ylabel("Evaluation Score", fontsize=11)
    plt.xlabel("")
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig6_path = "figures/06_model_comparison_metrics.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Generated: {fig6_path}")
    
    # -------------------------------------------------------------
    # Figure 7: ROC and Precision-Recall Curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 7A: ROC Curves
    for name in models.keys():
        y_prob = test_predictions[name]["prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.6)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[0].legend(loc="lower right")
    
    # 7B: Precision-Recall Curves
    baseline_churn_rate = y_test.mean()
    for name in models.keys():
        y_prob = test_predictions[name]["prob"]
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc_val = average_precision_score(y_test, y_prob)
        axes[1].plot(recall_vals, precision_vals, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", linewidth=2)
    axes[1].axhline(baseline_churn_rate, color="k", linestyle="--", label=f"Baseline Churn Rate ({baseline_churn_rate:.2f})", alpha=0.6)
    axes[1].set_title("Precision-Recall (PR) Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall (Coverage of Churners)", fontsize=11)
    axes[1].set_ylabel("Precision (Quality of Churn Alerts)", fontsize=11)
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    fig7_path = "figures/07_roc_and_pr_curves.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"Generated: {fig7_path}")
    
    # -------------------------------------------------------------
    # Figure 8: Confusion Matrices for All Models
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, clf) in enumerate(models.items()):
        y_pred = test_predictions[name]["pred"]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"]
        )
        axes[i].set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Predicted Label", fontsize=10)
        axes[i].set_ylabel("True Label", fontsize=10)
        
    plt.tight_layout()
    fig8_path = "figures/08_confusion_matrices.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"Generated: {fig8_path}")

if __name__ == "__main__":
    train_and_evaluate()

