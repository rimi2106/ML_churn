import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Drop irrelevant columns
df = df.drop(columns=['customerID'])

# Convert 'TotalCharges' to numeric, handling errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Replace "No internet service" and "No phone service" with "No"
cols_to_clean = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                 "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]
for col in cols_to_clean:
    df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

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

# User Inputs
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

# Create input dataframe
input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, contract, internet_service, paperless_billing, 
                            payment_method, phone_service, multiple_lines, online_security, online_backup, 
                            device_protection, tech_support, streaming_tv, streaming_movies, partner, dependents]],
                          columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaperlessBilling', 
                                   'PaymentMethod', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Partner', 'Dependents'])

# Convert categorical features to string type
for col in categorical_features:
    input_data[col] = input_data[col].astype(str)

# Convert numerical features to float and handle missing values
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

# Ensure all required columns exist in input data
missing_cols = set(X_train.columns) - set(input_data.columns)
for col in missing_cols:
    if col in categorical_features:
        input_data[col] = "Unknown"
    else:
        input_data[col] = 0

# Reorder columns to match training data
input_data = input_data[X_train.columns]

# Debugging: Display final input data before prediction
st.write("🔹 Final Processed Input Data for Model")
st.write(input_data)

# Predict
if st.button("Predict Churn"):
    st.write("🔹 Button Clicked! Running Prediction...")
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0]
        st.write("✅ Prediction Successful!")
        
        # Display results
        st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
        st.write("Churn Probability:", round(probability * 100, 2), "%")
    except Exception as e:
        st.write("❌ Error During Prediction:", str(e))
