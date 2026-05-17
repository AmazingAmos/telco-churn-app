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
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib  # Changed from pickle
import json

# Load data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Separate features and target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify numeric and categorical columns
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', LabelEncoder())  # Simplified for now
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_resampled, y_train_resampled)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)

# Save models using joblib (more reliable than pickle)
joblib.dump(lr_model, 'model/logistic_model.pkl')
joblib.dump(rf_model, 'model/rf_model.pkl')

# Save metadata
metadata = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'feature_names': list(X.columns),
    'preprocessing': 'ColumnTransformer with StandardScaler and LabelEncoder'
}

with open('model/model_metadata.pkl', 'wb') as f:
    joblib.dump(metadata, f)

print("✅ Models saved successfully!")
print(f"Logistic Regression Score: {lr_model.score(X_train_resampled, y_train_resampled):.4f}")
print(f"Random Forest Score: {rf_model.score(X_train_resampled, y_train_resampled):.4f}")
