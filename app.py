import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

# ======================================
# TITLE & DESCRIPTION
# ======================================
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict "
    "whether a loan will be approved by combining multiple ML models for better decision making."
)

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv(
    r"train_loan.csv"
)

df.drop(columns=["Loan_ID"], inplace=True)

cat_cols = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Property_Area"]

# ✅ FIX: include income columns
num_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ======================================
# SPLIT & SCALE
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ======================================
# TRAIN STACKING MODEL (ONCE)
# ======================================
base_models = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
]

meta_model = LogisticRegression(max_iter=1000)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=8
)

stacking_model.fit(X_train, y_train)

# ======================================
# INPUT SECTION (SIDEBAR)
# ======================================
st.sidebar.header("📥 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0.0, value=5000.0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0.0, value=2000.0)
loan_amt = st.sidebar.number_input("Loan Amount", min_value=0.0, value=150.0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0, value=360)

credit_hist = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# ======================================
# MODEL ARCHITECTURE DISPLAY
# ======================================
st.subheader("🧩 Stacking Model Architecture")
st.info(
    """
    **Base Models Used**
    - Logistic Regression
    - Decision Tree
    - Random Forest

    **Meta Model Used**
    - Logistic Regression
    """
)

# ======================================
# PREPARE USER INPUT
# ======================================
input_data = {
    "ApplicantIncome": app_income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amt,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1 if credit_hist == "Yes" else 0,
    "Self_Employed_Yes": 1 if employment == "Self-Employed" else 0,
    "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    "Property_Area_Semiurban": 1 if property_area == "Semi-Urban" else 0
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_df[num_cols] = scaler.transform(input_df[num_cols])

# ======================================
# PREDICTION BUTTON
# ======================================
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    final_pred = stacking_model.predict(input_df)[0]
    confidence = stacking_model.predict_proba(input_df)[0][1] * 100

    st.subheader("📊 Prediction Result")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write("Approved" if final_pred == 1 else "Rejected")

    st.markdown("### 📈 Confidence Score")
    st.write(f"{confidence:.2f}%")

    st.subheader("💼 Business Explanation")
    st.info(
        f"""
        Based on income, credit history, and combined predictions from multiple models,
        the applicant is **{'likely' if final_pred == 1 else 'unlikely'} to repay the loan**.

        Therefore, the **stacking model predicts loan {'approval' if final_pred == 1 else 'rejection'}**.
        """
    )
