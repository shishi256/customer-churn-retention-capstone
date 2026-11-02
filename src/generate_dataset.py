"""
Data Generation Script for Customer Churn & Retention Optimization Capstone Project.
Generates an authentic, industry-standard subscription dataset (N = 8,500)
incorporating realistic behavioural distributions, cross-feature correlations,
dirty real-world data patterns (e.g. whitespace missing values), and realistic churn propensities.
"""

import os
import numpy as np
import pandas as pd

def generate_customer_dataset(n_samples=8500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Customer Identifiers & Demographics
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.505, 0.495])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.838, 0.162])
    
    # Partner & Dependents correlation
    partner_prob = np.where(senior_citizen == 1, 0.45, 0.52)
    partner = np.where(np.random.rand(n_samples) < partner_prob, "Yes", "No")
    
    dependents_prob = np.where(partner == "Yes", 0.48, 0.12)
    dependents = np.where(np.random.rand(n_samples) < dependents_prob, "Yes", "No")
    
    # 2. Tenure (Distribution with peak at early tenure and spike at long tenure)
    # Bimodal mixture: 45% early/mid tenure, 35% mid tenure, 20% sticky long-term
    t1 = np.random.exponential(scale=12, size=n_samples)
    t2 = np.random.uniform(1, 72, size=n_samples)
    t3 = np.random.normal(loc=65, scale=5, size=n_samples)
    choice = np.random.choice([0, 1, 2], size=n_samples, p=[0.45, 0.35, 0.20])
    tenure_raw = np.where(choice == 0, t1, np.where(choice == 1, t2, t3))
    tenure_months = np.clip(np.round(tenure_raw), 0, 72).astype(int)
    
    # 3. Contract Type (strongly correlated with tenure)
    contract_probs = np.zeros((n_samples, 3))
    for i in range(n_samples):
        t = tenure_months[i]
        if t <= 12:
            p = [0.82, 0.14, 0.04]  # mostly month-to-month
        elif t <= 36:
            p = [0.45, 0.35, 0.20]
        else:
            p = [0.15, 0.35, 0.50]  # long term multi-year
        contract_probs[i] = p
        
    contract_types = ["Month-to-month", "One year", "Two year"]
    contract = [np.random.choice(contract_types, p=contract_probs[i]) for i in range(n_samples)]
    contract = np.array(contract)
    
    # 4. Core Phone and Internet Services
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.903, 0.097])
    multiple_lines = []
    for p in phone_service:
        if p == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
    multiple_lines = np.array(multiple_lines)
    
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.44, 0.34, 0.22])
    
    # Internet add-on services
    addon_features = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    addons = {}
    
    for feat in addon_features:
        feat_vals = []
        for inet in internet_service:
            if inet == "No":
                feat_vals.append("No internet service")
            else:
                prob = 0.40 if feat in ["OnlineSecurity", "TechSupport"] else 0.50
                feat_vals.append(np.random.choice(["Yes", "No"], p=[prob, 1 - prob]))
        addons[feat] = np.array(feat_vals)
        
    # 5. Billing & Payment
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    
    # Payment method distribution depends on contract & senior citizen
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    payment_method = []
    for i in range(n_samples):
        if contract[i] == "Month-to-month":
            p = [0.52, 0.18, 0.15, 0.15]
        else:
            p = [0.15, 0.15, 0.35, 0.35]
        payment_method.append(np.random.choice(payment_methods, p=p))
    payment_method = np.array(payment_method)
    
    # 6. Behavioral & Engagement Metrics
    # Support tickets: customers with issues contact support
    # Monthly login frequency: active users log in more
    # Customer satisfaction score (1 to 5)
    support_tickets = np.random.poisson(lam=1.4, size=n_samples)
    # Month to month users and fiber optic users without tech support experience more friction
    friction_boost = np.where((internet_service == "Fiber optic") & (addons["TechSupport"] == "No"), 1.2, 0.0)
    support_tickets = np.clip(np.round(support_tickets + friction_boost + np.random.exponential(scale=0.5, size=n_samples)), 0, 9).astype(int)
    
    login_frequency = np.clip(np.round(np.random.normal(loc=18, scale=6, size=n_samples) - 0.5 * support_tickets), 1, 45).astype(int)
    
    # Satisfaction score (1 = Very unsatisfied, 5 = Highly satisfied)
    satisfaction_latent = 4.0 - 0.45 * support_tickets + 0.02 * tenure_months - np.where(contract == "Month-to-month", 0.3, -0.2) + np.random.normal(0, 0.6, n_samples)
    satisfaction_score = np.clip(np.round(satisfaction_latent), 1, 5).astype(int)
    
    # 7. Pricing: Monthly Charges & Total Charges
    base_charge = 19.5
    monthly_charges = np.full(n_samples, base_charge)
    monthly_charges += np.where(phone_service == "Yes", 5.0, 0.0)
    monthly_charges += np.where(multiple_lines == "Yes", 8.5, 0.0)
    monthly_charges += np.where(internet_service == "DSL", 24.5, np.where(internet_service == "Fiber optic", 49.0, 0.0))
    for feat in addon_features:
        monthly_charges += np.where(addons[feat] == "Yes", 7.5, 0.0)
    # Add minor random variance in pricing plans/taxes
    monthly_charges += np.random.normal(loc=0.0, scale=1.75, size=n_samples)
    monthly_charges = np.round(np.clip(monthly_charges, 18.25, 119.50), 2)
    
    # Total charges: tenure * monthly_charges + noise
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 10, n_samples)
    total_charges = np.where(tenure_months == 0, 0.0, np.clip(total_charges, 0, None))
    total_charges = np.round(total_charges, 2)
    
    # 8. Churn Propensity Engine (Latent Logistic Model)
    # Formula calibrated to produce realistic churn (~26.5% overall churn rate)
    # Key industry drivers:
    # (+) Month-to-month contract, High monthly charges, Electronic check, High support tickets, Fiber optic without security
    # (-) High tenure, Tech support, Long term contract, High satisfaction score, Partner/Dependents
    logit = (
        -1.30
        + 1.45 * (contract == "Month-to-month")
        - 1.10 * (contract == "Two year")
        - 0.55 * (contract == "One year")
        - 0.042 * tenure_months
        + 0.016 * monthly_charges
        + 0.65 * (payment_method == "Electronic check")
        + 0.42 * (internet_service == "Fiber optic")
        - 0.52 * (addons["TechSupport"] == "Yes")
        - 0.40 * (addons["OnlineSecurity"] == "Yes")
        + 0.48 * support_tickets
        - 0.65 * satisfaction_score
        - 0.22 * (partner == "Yes")
        - 0.25 * (dependents == "Yes")
        + 0.28 * (paperless_billing == "Yes")
        + 0.20 * senior_citizen
    )
    
    prob_churn = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(np.random.rand(n_samples) < prob_churn, "Yes", "No")
    
    # 9. Introduce Real-World Imperfections for Data Cleaning Showcase
    # Introduce whitespace strings in TotalCharges for tenure == 0 (classic Telco quirk)
    total_charges_raw = [
        " " if tenure_months[i] == 0 and np.random.rand() < 0.9 else str(total_charges[i])
        for i in range(n_samples)
    ]
    
    # Construct DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure_months,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": addons["OnlineSecurity"],
        "OnlineBackup": addons["OnlineBackup"],
        "DeviceProtection": addons["DeviceProtection"],
        "TechSupport": addons["TechSupport"],
        "StreamingTV": addons["StreamingTV"],
        "StreamingMovies": addons["StreamingMovies"],
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_raw,
        "SupportTickets": support_tickets,
        "LoginFrequency": login_frequency,
        "SatisfactionScore": satisfaction_score,
        "Churn": churn
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_customer_dataset(n_samples=8500, random_state=42)
    output_path = "data/raw/customer_churn_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated raw dataset: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")
    print(f"Sample preview:\n{df.head(3)}")

