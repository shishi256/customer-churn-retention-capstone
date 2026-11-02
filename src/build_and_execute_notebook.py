"""
Notebook Generation and Automated Execution Module.
Constructs a comprehensive, publication-grade Jupyter Notebook:
'notebooks/capstone_churn_prediction.ipynb'
Covering Objectives 1 through 7 with rich Markdown narratives,
executed code cells, visualizations, tables, and business takeaways.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.12.0"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }
    
    cells = []
    
    # -------------------------------------------------------------------------
    # Title & Executive Summary
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# 📊 Data Science Capstone Project
## Customer Churn Prediction & Retention Optimization: An End-to-End Decision Intelligence Framework
**YuvaIntern Final Capstone Showcase**  
*Author:* Antigravity Data Science & Machine Learning Engineering  
*Domain:* Subscription Services / Customer Lifetime Value (CLV) Optimization

---

### Project Architecture & Objectives
1. **Define the Capstone Problem:** Formulate the business problem, churn economics, objectives, constraints, and success criteria.
2. **Collect & Prepare Data:** Ingestion, data hygiene, missing value handling, exploratory data analysis, and domain-specific feature engineering.
3. **Apply Data Science Techniques:** Parametric and non-parametric statistical hypothesis testing (Chi-square, Cramér's V, Welch's t-test, Cohen's d).
4. **Build & Evaluate the Solution:** 5-Fold Stratified Cross-Validation benchmarking of multiple ML algorithms (Logistic Regression, Random Forest, HistGradientBoosting, MLP Neural Network) with class imbalance treatment.
5. **Analyze Results, Errors & Limitations:** False Positive vs False Negative cost trade-offs, probability calibration curves, and optimal threshold selection via financial payoff matrix.
6. **Generate Insights & Recommendations:** Permutation feature importance, unsupervised customer persona segmentation, and concrete ROI retention strategy.
7. **Document & Present the Capstone:** Production modular codebase, interactive presentation deck, executive report, and real-time inference CLI."""))

    # -------------------------------------------------------------------------
    # Setup & Imports
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup & Library Ingestion"))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

# Add project root to system path
sys.path.append(os.path.abspath(".."))
from src.preprocessing import clean_data, engineer_features, build_preprocessor
from src.statistical_tests import run_categorical_tests, run_numerical_tests

# Visualization configuration
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("Environment verified. All modules imported successfully.")"""))

    # -------------------------------------------------------------------------
    # Objective 2: Data Collection & Cleaning
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Data Ingestion, Hygiene, & Preprocessing
Here we ingest the customer churn dataset, inspect structure, handle whitespace missing values in `TotalCharges`, and encode target variables."""))
    cells.append(nbf.v4.new_code_cell("""# Load raw dataset
raw_df = pd.read_csv("../data/raw/customer_churn_raw.csv")
print(f"Dataset Dimensions: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
raw_df.head(5)"""))

    cells.append(nbf.v4.new_code_cell("""# Clean data and inspect missing values
clean_df = clean_data(raw_df)
print("Data Types & Null Counts:")
print(clean_df.info())
print("\nTarget Class Distribution:")
print(clean_df['Churn'].value_counts(normalize=True).round(4) * 100)"""))

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""### 2.2 Advanced Feature Engineering
We construct domain-specific features capturing:
- `total_services`: Ecosystem breadth (stickiness indicator)
- `avg_cost_per_service`: Perceived price burden
- `ticket_velocity`: Support friction per unit tenure
- `is_high_risk_contract_payment`: High-friction cohort indicator
- `has_security_and_support`: Protective service bundle"""))
    cells.append(nbf.v4.new_code_cell("""processed_df = engineer_features(clean_df)
print(f"Feature Engineered Dimensions: {processed_df.shape[1]} columns")
processed_df[['customerID', 'tenure', 'MonthlyCharges', 'total_services', 'avg_cost_per_service', 'ticket_velocity', 'Churn']].head(5)"""))

    # -------------------------------------------------------------------------
    # Objective 2 & 3: EDA & Visualizations
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis & Statistical Inference
### 3.1 Exploratory Visualizations
Let's visualize key driver patterns including tenure attrition, contract vulnerability, support ticket escalation, and product bundle stickiness."""))
    cells.append(nbf.v4.new_code_cell("""# Display Figure 1: Tenure Dynamics
from IPython.display import Image, display
display(Image(filename="../figures/01_churn_distribution_and_tenure.png"))"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 2: Contract Horizon & Payment Methods
display(Image(filename="../figures/02_contract_and_billing_impact.png"))"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 3: Support Friction & Customer Satisfaction
display(Image(filename="../figures/03_support_and_satisfaction_friction.png"))"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 5: Numerical Correlation Heatmap
display(Image(filename="../figures/05_numerical_correlation_matrix.png"))"""))

    # -------------------------------------------------------------------------
    # Objective 3: Statistical Hypothesis Testing
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Formal Hypothesis Testing
We perform two-sided parametric and non-parametric hypothesis tests to evaluate associations with customer churn:
- **Chi-Square Test ($\chi^2$) & Cramér's V** for categorical features
- **Welch's t-test & Cohen's d** for continuous features"""))
    cells.append(nbf.v4.new_code_cell("""cat_stats = run_categorical_tests(processed_df)
print("Top Categorical Associations by Cramér's V:")
display(cat_stats.head(8))"""))

    cells.append(nbf.v4.new_code_cell("""num_stats = run_numerical_tests(processed_df)
print("Continuous Feature Differences by Cohen's d Effect Size:")
display(num_stats.head(8))"""))

    # -------------------------------------------------------------------------
    # Objective 4: Machine Learning Modeling & Benchmarking
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Machine Learning Modeling & Multi-Model Benchmarking
We benchmark 4 distinct model families using 5-Fold Stratified Cross-Validation on the training split and evaluate out-of-sample performance on the holdout test set (20%):
1. **Logistic Regression (L2 Regularized, Balanced Weights)**
2. **Random Forest Classifier (200 Trees)**
3. **HistGradientBoosting Classifier**
4. **Multi-Layer Perceptron (Neural Network)**"""))
    cells.append(nbf.v4.new_code_cell("""bench_results = pd.read_csv("../reports/model_benchmark_results.csv")
print("=== MODEL BENCHMARK SUMMARY ===")
display(bench_results)"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 6: Model Comparison Bar Chart
display(Image(filename="../figures/06_model_comparison_metrics.png"))"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 7: ROC and Precision-Recall Curves
display(Image(filename="../figures/07_roc_and_pr_curves.png"))"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 8: Confusion Matrices across all models
display(Image(filename="../figures/08_confusion_matrices.png"))"""))

    # -------------------------------------------------------------------------
    # Objective 5: Error Analysis, Calibration & Threshold Optimization
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 5. Diagnostic Error Analysis & Financial Threshold Optimization
Standard machine learning models use an arbitrary 0.50 probability threshold. In customer churn, **False Negatives** (losing a customer worth \$720) are far more punitive than **False Positives** (sending an \$65 proactive offer).

Here we optimize the decision threshold against an asymmetric financial cost-benefit payoff matrix."""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename="../figures/09_expected_profit_vs_threshold.png"))"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename="../figures/10_calibration_curves.png"))"""))

    cells.append(nbf.v4.new_code_cell("""# Error analysis profile breakdown
err_profiles = pd.read_csv("../reports/error_analysis_profiles.csv")
display(err_profiles)"""))

    # -------------------------------------------------------------------------
    # Objective 6: Explainability & Persona Clustering
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 6. Model Explainability & Strategic Customer Personas
### 6.1 Permutation Feature Importance
Understanding which features drive the model's predictive power."""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename="../figures/11_feature_importance.png"))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 6.2 At-Risk Customer Persona Clustering
Using unsupervised K-Means on the predicted high-risk population, we derive 3 distinct actionable customer personas."""))
    cells.append(nbf.v4.new_code_cell("""personas_df = pd.read_csv("../reports/customer_personas_summary.csv")
display(personas_df)"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename="../figures/12_customer_risk_personas.png"))"""))

    # -------------------------------------------------------------------------
    # Objective 6: Business Recommendations & ROI
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 7. Business Recommendations & Projected Financial ROI

### Strategic Recommendations:
1. **Persona A (High-Friction Month-to-Month):**
   - **Intervention:** Proactive customer success outreach within 48 hours when support tickets $\ge 2$.
   - **Incentive:** 15% discount for converting from Month-to-month to Annual contract with automated payment.
2. **Persona B (Price-Sensitive Streamers):**
   - **Intervention:** Loyalty bundle discount combining Streaming TV with free 6-month TechSupport and Cybersecurity.
3. **Persona C (Frustrated Onboarders):**
   - **Intervention:** Dedicated onboarding specialist check-in during Day 14 and Day 30 to troubleshoot fiber optic installation friction.

### Financial Return on Investment (ROI) Projection:
- **Baseline Churn Rate:** 24.9% across 8,500 customer base = 2,116 lost customers/year.
- **Gross Revenue at Risk:** 2,116 × \$720 annual LTV = **\$1,523,520**.
- **Model-Targeted Cohort:** Targeting at optimal threshold (0.13) reaches 94.8% of true churners.
- **Intervention Success (50% save rate):** 1,003 customers retained.
- **Annual Gross Revenue Saved:** 1,003 × \$720 = **\$722,160**.
- **Total Campaign Outlay:** \$65 × 1,780 targeted customers = **\$115,700**.
- **Net Annual Bottom-Line Profit:** **\$606,460** (ROI of **524%**)."""))

    # -------------------------------------------------------------------------
    # Objective 7: Interactive Prediction Demo
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 8. Real-Time Inference Demo
We test the production pipeline on an individual customer profile."""))
    cells.append(nbf.v4.new_code_cell("""from predict import score_customer

sample_profile = {
    "customerID": "INFERENCE-DEMO-01",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 84.50,
    "TotalCharges": 169.00,
    "SupportTickets": 3,
    "LoginFrequency": 14,
    "SatisfactionScore": 2
}

scoring_result = score_customer(sample_profile, optimal_threshold=0.13)
for k, v in scoring_result.items():
    print(f"{k.upper():<25}: {v}")"""))

    nb.cells = cells
    
    os.makedirs("notebooks", exist_ok=True)
    nb_path = "notebooks/capstone_churn_prediction.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook blueprint: {nb_path}")
    
    # Execute notebook
    print("Executing notebook end-to-end to save output cells...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_to_run = nbf.read(f, as_version=4)
        
    client = NotebookClient(nb_to_run, timeout=600, kernel_name="python3")
    client.execute()
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb_to_run, f)
    print("Notebook executed and saved successfully with full cell outputs!")

if __name__ == "__main__":
    build_notebook()

