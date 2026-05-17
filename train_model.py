"""
Model Training Script for Telco Churn Prediction
Extracts the training logic from the original notebook and saves models
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🔄 Loading data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data preprocessing
print("🔄 Preprocessing data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Prepare features
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
print("🔄 Balancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Train Logistic Regression
print("🔄 Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_balanced, y_train_balanced)

# Train Random Forest
print("🔄 Training Random Forest...")
rf_model = RandomForestClassifier(
    random_state=42, 
    n_estimators=100, 
    class_weight='balanced',
    max_depth=10
)
rf_model.fit(X_train_balanced, y_train_balanced)

# Save models
print("💾 Saving models...")
joblib.dump(lr_model, 'logistic_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names and metadata
metadata = {
    'feature_names': list(X.columns),
    'categorical_columns': list(categorical_cols),
    'numeric_columns': ['tenure', 'MonthlyCharges', 'TotalCharges']
}
joblib.dump(metadata, 'model_metadata.pkl')

# Evaluate models
print("\n📊 Model Performance:")
print(f"Logistic Regression Train Score: {lr_model.score(X_train_balanced, y_train_balanced):.4f}")
print(f"Random Forest Train Score: {rf_model.score(X_train_balanced, y_train_balanced):.4f}")

print("\n✅ Models saved successfully!")
print("Files created:")
print("  - logistic_model.pkl")
print("  - rf_model.pkl")
print("  - scaler.pkl")
print("  - model_metadata.pkl")
