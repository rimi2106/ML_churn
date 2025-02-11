import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from flask import Flask, request, jsonify
# Load Dataset (Replace with actual dataset path)
df = pd.read_csv("Telco-Customer-Churn.csv")
df.head(10)
# Drop irrelevant columns
df = df.drop(columns=['customerID'])

# Convert 'TotalCharges' to numeric, handling errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
# Split features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Save Model
joblib.dump(model, "telco_churn_model.pkl")

# Streamlit Dashboard
st.title("Telco Customer Churn Prediction Dashboard")

# Input Fields
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Create input dataframe
input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, contract, internet_service, paperless_billing]],
                          columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaperlessBilling'])

# Load Model
model = joblib.load("telco_churn_model.pkl")

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1][0]
    st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
    st.write("Churn Probability:", round(probability * 100, 2), "%")
