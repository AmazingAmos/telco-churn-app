"""
Model Training Script for Telco Churn Prediction
Extracts the training logic from the original notebook and saves models
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("TELCO CUSTOMER CHURN - MODEL TRAINING")
print("=" * 60)

# 1) Load data
print("\n[1/6] Loading dataset...")
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Dataset not found at {DATA_PATH}")
    print("\nPlease download the dataset:")
    print("   URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("   Or: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113")
    print("\nSave it as 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the same directory.")
    exit(1)

df_raw = pd.read_csv(DATA_PATH)
print(f"✓ Dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# 2) Data Preprocessing
print("\n[2/6] Preprocessing data...")

# Drop customerID (not a feature)
df = df_raw.drop('customerID', axis=1, errors='ignore')

# Fix TotalCharges (convert to numeric, handle whitespace)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges
# If tenure=0 and TotalCharges is missing, set to 0
# Otherwise fill with median
df.loc[(df['tenure'] == 0) & (df['TotalCharges'].isna()), 'TotalCharges'] = 0.0
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert SeniorCitizen to string for consistency
df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

print(f"✓ Data cleaned: {df.isna().sum().sum()} missing values remaining")

# 3) Feature Engineering
print("\n[3/6] Splitting features and target...")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify numeric and categorical columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in X.columns if col not in numeric_cols]

print(f"✓ Numeric features: {len(numeric_cols)}")
print(f"✓ Categorical features: {len(categorical_cols)}")

# 4) Train-Test Split
print("\n[4/6] Creating train-test split (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# 5) Build and Train Models
print("\n[5/6] Training models...")

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Model 1: Balanced Logistic Regression (Recommended)
print("\n  Training Balanced Logistic Regression...")
lr_balanced = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        solver='lbfgs'
    ))
])
lr_balanced.fit(X_train, y_train)

# Evaluate on test set
lr_score = lr_balanced.score(X_test, y_test)
print(f"  ✓ Accuracy: {lr_score:.4f}")

# Model 2: Random Forest with balanced subsample
print("\n  Training Random Forest Classifier...")
rf_model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])
rf_model.fit(X_train, y_train)

# Evaluate on test set
rf_score = rf_model.score(X_test, y_test)
print(f"  ✓ Accuracy: {rf_score:.4f}")

# 6) Save Models
print("\n[6/6] Saving trained models...")

with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(lr_balanced, f)
print("  ✓ Saved: logistic_model.pkl")

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("  ✓ Saved: rf_model.pkl")

# Save feature lists for reference
model_metadata = {
    'numeric_features': numeric_cols,
    'categorical_features': categorical_cols,
    'all_features': list(X.columns),
    'target': 'Churn',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'lr_accuracy': lr_score,
    'rf_accuracy': rf_score,
    'random_state': RANDOM_STATE
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)
print("  ✓ Saved: model_metadata.pkl")

# Print summary
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nModel Performance Summary:")
print(f"  • Balanced Logistic Regression: {lr_score:.2%} accuracy")
print(f"  • Random Forest: {rf_score:.2%} accuracy")
print("\nFiles created:")
print("  • logistic_model.pkl (Recommended for deployment)")
print("  • rf_model.pkl")
print("  • model_metadata.pkl")
print("\nNext steps:")
print("  1. Run: streamlit run app.py")
print("  2. Test the app locally")
print("  3. Deploy to Streamlit Community Cloud")
print("=" * 60)
