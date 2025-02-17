import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Select only relevant features
selected_features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaperlessBilling", "PaymentMethod"]
df = df[selected_features + ['Churn']]

# Convert 'TotalCharges' to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify categorical and numerical features
categorical_features = ["Contract", "InternetService", "PaperlessBilling", "PaymentMethod"]
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "telco_churn_model_minimal.pkl")

# Streamlit Dashboard
st.title("Telco Customer Churn Prediction")

# User Inputs
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Create input dataframe
input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, contract, internet_service, paperless_billing, payment_method]],
                          columns=["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaperlessBilling", "PaymentMethod"])

# Convert categorical features to string type
for col in categorical_features:
    input_data[col] = input_data[col].astype(str)

# Convert numerical features to float and handle missing values
for col in numerical_features:
    input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

# Load Model
model = joblib.load("telco_churn_model_minimal.pkl")

# Predict
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0]
        st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
        st.write("Churn Probability:", round(probability * 100, 2), "%")
    except Exception as e:
        st.write("❌ Error During Prediction:", str(e))
