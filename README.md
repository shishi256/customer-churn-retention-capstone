# 🎓 Customer Churn Prediction & Retention Optimization
### YuvaIntern Final Capstone Project: End-to-End Data Science & Machine Learning Showcase

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary
This project showcases an industry-standard, end-to-end Data Science Capstone addressing **Subscription Customer Churn and Retention Optimization**. Developed to fulfill all 7 core Capstone Objectives, the framework combines **rigorous parametric and non-parametric statistical hypothesis testing**, **multi-architecture machine learning benchmarking (0.932 Test ROC-AUC)**, **asymmetric financial decision threshold optimization**, **explainability and customer persona clustering**, and **an automated real-time inference CLI**.

---

## 🏆 Key Achievements & Business Metrics
- **Model Accuracy & Sensitivity:** **0.932 Test ROC-AUC**, **0.815 PR-AUC**, and **87.0% Recall** on the minority churn class.
- **Financial Threshold Optimization:** Maximized expected net campaign profit by adjusting classification cutoff from default 0.50 to **0.13**, capturing **94.8% of all churners** and delivering a **+6.9% financial lift**.
- **Enterprise Financial ROI:** In an enterprise base of 8,500 subscribers, the framework preserves **1,003 customers annually**, saving **$722,160 in recurring revenue** against a campaign cost of $115,700—generating **$606,460 in net profit (524% ROI)**.

---

## 📂 Repository Structure
```
yuvaintern final/
├── README.md                              <- Project overview, quickstart & reproduction guide
├── REPORT.md                              <- Comprehensive academic & executive capstone report
├── PRESENTATION.html                      <- Interactive visual presentation deck (open in browser)
├── PRESENTATION.md                        <- Markdown presentation slides
├── predict.py                             <- Real-time CLI scoring & customer decision engine
├── data/
│   ├── raw/customer_churn_raw.csv         <- Raw ingested subscription dataset (N = 8,500)
│   └── processed/customer_churn_processed.csv <- Cleaned and feature engineered dataset
├── notebooks/
│   └── capstone_churn_prediction.ipynb    <- Publication-grade Jupyter Notebook
├── src/
│   ├── generate_dataset.py                <- Realistic synthetic subscription dataset generator
│   ├── preprocessing.py                   <- Data hygiene, feature engineering & pipelines
│   ├── statistical_tests.py               <- Chi-Square, t-tests, Cramér's V, Cohen's d
│   ├── train_models.py                    <- 5-Fold Stratified CV model benchmarking
│   ├── error_and_threshold_analysis.py    <- Cost-benefit optimization & calibration curves
│   └── explainability_and_personas.py     <- Permutation importance & K-Means personas
├── figures/                               <- 12 High-resolution publication visual artifacts
│   ├── 01_churn_distribution_and_tenure.png
│   ├── 02_contract_and_billing_impact.png
│   ├── 03_support_and_satisfaction_friction.png
│   ├── 04_financial_load_and_services.png
│   ├── 05_numerical_correlation_matrix.png
│   ├── 06_model_comparison_metrics.png
│   ├── 07_roc_and_pr_curves.png
│   ├── 08_confusion_matrices.png
│   ├── 09_expected_profit_vs_threshold.png
│   ├── 10_calibration_curves.png
│   ├── 11_feature_importance.png
│   └── 12_customer_risk_personas.png
├── models/                                <- Serialized production model artifacts
│   ├── preprocessor.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── histgradientboosting.joblib
│   ├── random_forest.joblib
│   └── neural_net_mlp.joblib
└── reports/                               <- Exported CSV benchmark metrics
    ├── model_benchmark_results.csv
    ├── threshold_optimization.csv
    ├── error_analysis_profiles.csv
    ├── feature_importance.csv
    └── customer_personas_summary.csv
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone or open the workspace
cd "yuvaintern final"

# Use virtual environment with Python 3.12+
source .venv/bin/activate

# Install required packages (if not already installed)
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyter
```

### 2. Run the Full End-to-End Pipeline
Execute each pipeline component sequentially:
```bash
# 1. Generate Raw Data
python3 src/generate_dataset.py

# 2. Preprocess & Engineer Features
python3 src/preprocessing.py

# 3. Execute Statistical Tests
python3 src/statistical_tests.py

# 4. Generate EDA Visualizations
python3 src/eda_visualizations.py

# 5. Train & Benchmark ML Models
python3 src/train_models.py

# 6. Run Diagnostic Error Analysis & Financial Threshold Optimization
python3 src/error_and_threshold_analysis.py

# 7. Run Explainability & Customer Persona Clustering
python3 src/explainability_and_personas.py
```

### 3. Real-Time Customer Scoring CLI
Test real-time scoring on a live customer profile:
```bash
python3 predict.py
```
Output:
```json
{
  "customerID": "DEMO-9999",
  "churn_probability": 0.9734,
  "predicted_churn": true,
  "risk_tier": "Critical Risk",
  "recommended_action": "Priority Concierge Outreach + Annual Contract Discount Offer (20% off)"
}
```

### 4. Interactive Presentation Deck
Open `PRESENTATION.html` directly in any web browser to view the animated presentation slides with navigation controls, embedded charts, and speaker notes.

---

## 🎯 Alignment with Capstone Objectives

| # | Objective | Project Implementation & Evidence |
| :--- | :--- | :--- |
| **1** | **Define Capstone Problem** | Addressed in `REPORT.md` (Section 1) and `PRESENTATION.html`: Subscription churn economics, CAC vs retention cost, KPIs, and technical scope. |
| **2** | **Collect & Prepare Data** | Addressed in `src/generate_dataset.py` & `src/preprocessing.py`: 8,500 customer records, data cleaning, missing value handling, and domain feature engineering. |
| **3** | **Apply Data Science Techniques** | Addressed in `src/statistical_tests.py` & `src/eda_visualizations.py`: Chi-Square, Cramér's V, Welch's t-test, Cohen's d, and 5 EDA figures. |
| **4** | **Build & Evaluate Solution** | Addressed in `src/train_models.py`: 5-Fold Stratified CV on 4 model families (Logistic Regression, Random Forest, HistGradientBoosting, MLP). |
| **5** | **Analyze Results & Errors** | Addressed in `src/error_and_threshold_analysis.py`: Payoff matrix threshold optimization ($p^* = 0.13$), calibration curves, and error decomposition. |
| **6** | **Generate Insights & Strategy** | Addressed in `src/explainability_and_personas.py`: Permutation importance, 3 K-Means personas, and $606K net profit / 524% ROI retention strategy. |
| **7** | **Document & Present** | Comprehensive documentation via `REPORT.md`, `PRESENTATION.html`, `PRESENTATION.md`, `notebooks/capstone_churn_prediction.ipynb`, and `predict.py`. |

---
*Developed for the YuvaIntern Final Capstone Project Showcase.*

