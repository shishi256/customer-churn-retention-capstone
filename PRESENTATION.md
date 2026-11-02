# 📊 Capstone Presentation Slides: Customer Churn & Retention Optimization
*YuvaIntern Final Project Presentation*

---

## Slide 1: Title & Executive Snapshot
### Customer Churn Prediction & Retention Optimization: An End-to-End Decision Intelligence Framework
- **Domain:** Subscription / SaaS Customer Lifetime Value (CLV) Optimization
- **Key Results:**
  - **0.932 Test ROC-AUC** across 5-Fold Stratified Cross-Validation
  - **87.0% Churn Sensitivity (Recall)** on Holdout Test Split
  - **$606,460 Net Annual Bottom-Line Profit** (Targeting 8,500 subscriber base)
  - **524% Return on Retention Investment (ROI)**
- **Coverage:** Full fulfillment of Capstone Objectives 1 through 7.

---

## Slide 2: Problem Formulation & Business Economics (Objective 1)
- **The Problem:** Reactive customer retention fails (< 15% save rate after cancellation notice).
- **The Financial Stake:** 24.9% baseline churn rate = 2,116 lost customers = **$1.52 Million lost annually**.
- **The Asymmetric Payoff:** Missing an at-risk subscriber costs $720 in lost LTV. Sending an unnecessary retention offer costs only $65.
- **Project Objectives:**
  1. Statistically validate root causes of subscriber defection.
  2. Train and benchmark multi-architecture ML pipelines.
  3. Formulate an asymmetric decision threshold for maximal profit.
  4. Derive actionable customer personas with tailored interventions.

---

## Slide 3: Exploratory Data Analysis & Churn Dynamics (Objective 2)
- **The 12-Month Cliff:** Over 52% of all churn events occur within the first year. Median tenure for churners is 12.0 months vs. 37.5 months for retained users.
- **Contract Type Risk:**
  - Month-to-month contracts: **43.8% churn rate**.
  - One-year contracts: **11.2% churn rate**.
  - Two-year contracts: **2.9% churn rate**.
- **Payment Method Risk:** Electronic check users churn at **45.2%**, compared to < 15.5% for automatic bank/card payments.
- **Support Burden:** Churn escalates rapidly from 14.8% (0–1 tickets) to 58.2% (3+ tickets).
- *Visual Artifacts:* `figures/01_churn_distribution_and_tenure.png`, `figures/02_contract_and_billing_impact.png`.

---

## Slide 4: Statistical Hypothesis Testing (Objective 3)
- **Categorical Associations (Chi-Square & Cramér's V):**
  - **Contract Type:** $\chi^2 = 2,179.5$, $p < 10^{-100}$, Cramér's $V = \mathbf{0.5062}$ (Very strong association).
  - **Payment Method:** $\chi^2 = 668.5$, $p = 1.39 \times 10^{-144}$, Cramér's $V = \mathbf{0.2798}$ (Moderate association).
  - **Tech Support:** $\chi^2 = 166.1$, $p = 8.38 \times 10^{-37}$, Cramér's $V = \mathbf{0.1390}$ (Significant protective effect).
- **Continuous Differences (Welch's t-test & Cohen's d):**
  - **Satisfaction Score:** $t = -58.67$, $p < 10^{-100}$, Cohen's $d = \mathbf{-1.458}$ (Very large negative effect).
  - **Tenure:** $t = -58.55$, $p < 10^{-100}$, Cohen's $d = \mathbf{-1.141}$ (Large negative effect).
  - **Support Ticket Velocity:** $t = +26.83$, $p = 1.97 \times 10^{-138}$, Cohen's $d = \mathbf{+0.988}$ (Large positive effect).
- *Conclusion:* All core drivers demonstrate statistical significance at $\alpha = 0.001$.

---

## Slide 5: Machine Learning Solution Benchmarking (Objective 4)
- **Evaluation on 20% Holdout Split (1,700 customers):**
  1. **Logistic Regression (Baseline):** ROC-AUC **0.9322** | PR-AUC **0.8149** | Recall **87.03%** | F2 **0.8156**
  2. **Neural Network (MLP):** ROC-AUC 0.9278 | PR-AUC 0.8047 | Accuracy **87.00%** | Brier Score **0.0917**
  3. **HistGradientBoosting:** ROC-AUC 0.9191 | PR-AUC 0.7789 | Recall 79.72% | F2 0.7699
  4. **Random Forest (200 Trees):** ROC-AUC 0.9180 | PR-AUC 0.7820 | Recall 79.25% | F2 0.7678
- *Champion Selection:* Logistic Regression delivers top sensitivity on churners; MLP delivers superior probability calibration.
- *Visual Artifacts:* `figures/06_model_comparison_metrics.png`, `figures/07_roc_and_pr_curves.png`, `figures/08_confusion_matrices.png`.

---

## Slide 6: Diagnostic Error Analysis & Financial Threshold Optimization (Objective 5)
- **The Default 0.50 Flaw:** Arbitrary 0.50 threshold yields $89,245 in test profit, leaving 20.3% of churners unidentified.
- **Payoff Matrix:**
  - True Positive Gain: $+0.50 \times \$720 - \$65 = \mathbf{+\$295}$.
  - False Positive Cost: $\mathbf{-\$65}$ (Wasted promotional discount).
- **Optimal Cutoff:** Sweeping the curve identifies **$p^* = 0.13$** as optimal:
  - Test Set Net Profit: **$95,385 (+6.9% lift)**.
  - Test Set Churn Recall: **94.8%** (402 of 424 churners captured; only 22 missed).
- *Visual Artifacts:* `figures/09_expected_profit_vs_threshold.png`, `figures/10_calibration_curves.png`.

---

## Slide 7: Explainability & Customer Persona Segmentation (Objective 6)
- **Top 3 Permutation Drivers:** Contract Horizon, Ticket Velocity, Satisfaction Score.
- **Unsupervised K-Means Personas:**
  1. **Persona A: Frustrated Onboarders (High Friction):**
     - High tickets (3.7), high charges ($93.7/mo).
     - *Action:* Tier-3 concierge resolution call within 48h + billing credit.
  2. **Persona B: Price-Sensitive Streamers (Zero Stickiness):**
     - Low tenure (7.8 mo), single-line ($35/mo), no addon services.
     - *Action:* Bundle incentive (Free 6-month cybersecurity & backup).
  3. **Persona C: Month-to-Month High Friction:**
     - Electronic check, short tenure (6.9 mo), low satisfaction.
     - *Action:* 15% discount for 1-year annual contract + Autopay signup.
- *Visual Artifacts:* `figures/11_feature_importance.png`, `figures/12_customer_risk_personas.png`.

---

## Slide 8: Enterprise Business ROI & Deliverables (Objective 7)
- **Financial Projections (8,500 Customer Pool):**
  - Churn volume without ML: 2,116 customers lost ($1.52M revenue).
  - Model identified cohort: 1,780 targeted subscribers.
  - Successfully retained: **1,003 customers saved** (50% save rate).
  - Preserved annual recurring revenue: **$722,160**.
  - Total campaign cost: **$115,700** ($65/outreach).
  - **Net Bottom-Line Annual Savings:** **$606,460**.
  - **Retention Campaign ROI:** **524%**.
- **Project Deliverables:**
  - Production Codebase: `src/` modules.
  - Interactive Notebook: `notebooks/capstone_churn_prediction.ipynb`.
  - Executive Report: `REPORT.md`.
  - Interactive Deck: `PRESENTATION.html`.
  - CLI Inference Engine: `predict.py`.

