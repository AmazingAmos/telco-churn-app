import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all models and metadata"""
    try:
        lr_model = joblib.load('logistic_model.pkl')
        rf_model = joblib.load('rf_model.pkl')
       # scaler = joblib.load('scaler.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return lr_model, rf_model, scaler, metadata
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please ensure all model files are in the repository root.")
        return None, None, None

# Load models
lr_model, rf_model, metadata = load_models()

if lr_model is None:
    st.stop()

# App title
st.title("📊 Telco Customer Churn Prediction System")
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Customer Information")

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Balanced Logistic Regression", "Balanced Random Forest"],
    help="Choose which model to use for prediction"
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Demographic Info")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

st.sidebar.markdown("---")
st.sidebar.subheader("📞 Service Details")

phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Billing Info")

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.01)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 828.0, 0.01)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.info("""
    ### How to use this app:
    1. Fill in customer information in the **sidebar** on the left
    2. Click the **"Predict Churn Risk"** button below
    3. Review the prediction and recommendations
    """)

with col2:
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        categorical_cols = input_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on all possible values to avoid unseen labels
            if col in ['gender']:
                le.fit(['Male', 'Female'])
            elif col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                le.fit(['No', 'Yes'])
            elif col == 'MultipleLines':
                le.fit(['No', 'Yes', 'No phone service'])
            elif col == 'InternetService':
                le.fit(['DSL', 'Fiber optic', 'No'])
            elif col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                le.fit(['No', 'Yes', 'No internet service'])
            elif col == 'Contract':
                le.fit(['Month-to-month', 'One year', 'Two year'])
            elif col == 'PaymentMethod':
                le.fit(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            
            input_df[col] = le.transform(input_df[col])
        
        # Scale features
       # input_scaled = scaler.transform(input_df)
        
        # Make prediction
        model = lr_model if "Logistic" in selected_model else rf_model
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK**: This customer is likely to churn")
            st.metric("Churn Probability", f"{probability[1]:.1%}")
        else:
            st.success(f"✅ **LOW RISK**: This customer is likely to stay")
            st.metric("Retention Probability", f"{probability[0]:.1%}")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 1:
            st.markdown("""
            - **Offer retention incentives** (discounts, upgrades)
            - **Improve customer service** engagement
            - **Review contract terms** and pricing
            - **Address service quality** issues
            """)
        else:
            st.markdown("""
            - **Maintain service quality**
            - **Continue engagement** programs
            - **Monitor satisfaction** regularly
            """)

# About section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    This application uses machine learning to predict customer churn for a telecommunications company.
    
    **Models Used:**
    - Balanced Logistic Regression
    - Balanced Random Forest
    
    **Features:**
    - Real-time predictions
    - Risk assessment
    - Actionable recommendations
    """)
