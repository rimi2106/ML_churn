import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Upload dataset
st.title("Universal Customer Churn Prediction App")
uploaded_file = st.file_uploader("Telco-Customer-Churn", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Select target column
    target_col = st.selectbox("Select the target (churn) column", df.columns)

    # Auto-detect feature types
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = df.select_dtypes(exclude=["object"]).columns.tolist()

    # Remove the target column from features
    if target_col in categorical_features:
        categorical_features.remove(target_col)
    if target_col in numerical_features:
        numerical_features.remove(target_col)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].map({'Yes': 1, 'No': 0}) if df[target_col].dtype == 'object' else df[target_col]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    # Save Model
    joblib.dump(model, "churn_model.pkl")

    st.success("Model trained successfully! You can now make predictions.")

    # Make predictions on new inputs
    st.header("Make a Prediction")
    input_data = {}
    for col in X_train.columns:
        if col in numerical_features:
            input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        else:
            input_data[col] = st.selectbox(f"Select {col}", df[col].unique())

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    # Ensure feature alignment
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X_train.columns]

    if st.button("Predict Churn"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1][0]
        st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
        st.write("Churn Probability:", round(probability * 100, 2), "%")
