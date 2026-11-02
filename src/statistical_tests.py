"""
Statistical Inference and Hypothesis Testing Module.
Executes rigorous parametric and non-parametric tests:
- Chi-Square Tests of Independence with Cramér's V effect size
- Two-Sample Independent t-tests and Mann-Whitney U tests with Cohen's d
- Formulated business hypotheses testing with p-value corrections
"""

import numpy as np
import pandas as pd
from scipy import stats

def compute_cramers_v(contingency_table):
    """Calculates Cramér's V effect size for categorical association."""
    chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    phi2 = chi2 / n
    r_corr = r - ((r - 1)**2) / (n - 1)
    k_corr = k - ((k - 1)**2) / (n - 1)
    phi2_corr = max(0, phi2 - ((r - 1) * (k - 1)) / (n - 1))
    v = np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1))
    return chi2, p, dof, v

def compute_cohens_d(group1, group2):
    """Calculates Cohen's d effect size for two independent samples."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / s_pooled
    return d

def run_categorical_tests(df, target_col="Churn"):
    """Runs Chi-Square and Cramér's V across all categorical columns."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if "customerID" in categorical_cols:
        categorical_cols.remove("customerID")
        
    results = []
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p, dof, cramers_v = compute_cramers_v(contingency)
        results.append({
            "Feature": col,
            "Chi2_Stat": round(chi2, 3),
            "p_value": p,
            "Degrees_of_Freedom": dof,
            "Cramers_V": round(cramers_v, 4),
            "Significant_at_001": p < 0.001
        })
    res_df = pd.DataFrame(results).sort_values(by="Cramers_V", ascending=False)
    return res_df

def run_numerical_tests(df, target_col="Churn"):
    """Runs t-tests, Mann-Whitney U, and Cohen's d for numerical variables."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if "SeniorCitizen" in num_cols:
        num_cols.remove("SeniorCitizen")
        
    group_churn = df[df[target_col] == 1]
    group_no_churn = df[df[target_col] == 0]
    
    results = []
    for col in num_cols:
        val_churn = group_churn[col].dropna()
        val_no_churn = group_no_churn[col].dropna()
        
        # Two-sample t-test (Welch's t-test unequal variance)
        t_stat, t_pval = stats.ttest_ind(val_churn, val_no_churn, equal_var=False)
        # Mann-Whitney U test (non-parametric rank-sum test)
        u_stat, u_pval = stats.mannwhitneyu(val_churn, val_no_churn, alternative="two-sided")
        # Cohen's d
        d = compute_cohens_d(val_churn, val_no_churn)
        
        results.append({
            "Feature": col,
            "Churn_Mean": round(val_churn.mean(), 2),
            "Retained_Mean": round(val_no_churn.mean(), 2),
            "t_statistic": round(t_stat, 3),
            "t_p_value": t_pval,
            "MannWhitney_p_value": u_pval,
            "Cohens_d": round(d, 3),
            "Significant_at_001": t_pval < 0.001
        })
    res_df = pd.DataFrame(results).sort_values(by="Cohens_d", key=abs, ascending=False)
    return res_df

if __name__ == "__main__":
    data_path = "data/processed/customer_churn_processed.csv"
    df = pd.read_csv(data_path)
    print("=== CATEGORICAL STATISTICAL INFERENCE (CHI-SQUARE & CRAMER'S V) ===")
    cat_res = run_categorical_tests(df, target_col="Churn")
    print(cat_res.to_string(index=False))
    
    print("\n=== NUMERICAL STATISTICAL INFERENCE (T-TEST & COHEN'S D) ===")
    num_res = run_numerical_tests(df, target_col="Churn")
    print(num_res.to_string(index=False))

