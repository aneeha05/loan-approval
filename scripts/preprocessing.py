"""
Loan Approval Prediction - Step 2: Data Preprocessing
Purpose: Handle missing values, encode categorical features, prepare data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA AND SUMMARY
# ============================================================================
print("="*80)
print("LOAN APPROVAL PREDICTION - DATA PREPROCESSING")
print("="*80)

# Load dataset
df = pd.read_csv('loan_approval_dataset.csv')

# Load EDA summary
with open('data/eda_summary.json', 'r') as f:
    summary = json.load(f)

target_col = summary['target_column']
numerical_features = summary['numerical_features']
categorical_features = summary['categorical_features']

print(f"\nDataset Shape: {df.shape}")
print(f"Target Column: {target_col}")
print(f"Numerical Features: {len(numerical_features)}")
print(f"Categorical Features: {len(categorical_features)}")

# ============================================================================
# 2. HANDLE MISSING VALUES
# ============================================================================
print("\n1. Handling Missing Values")
print("-" * 80)

# Create a copy for processing
df_processed = df.copy()

# Check for missing values
missing_before = df_processed.isnull().sum()
total_missing = missing_before.sum()

if total_missing > 0:
    print(f"Total missing values: {total_missing}")
    print("\nMissing values by column:")
    print(missing_before[missing_before > 0])
    
    # Separate features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Handle numerical features - use median (robust to outliers)
    numerical_cols_with_missing = [col for col in numerical_features if X[col].isnull().any()]
    if numerical_cols_with_missing:
        print(f"\nImputing {len(numerical_cols_with_missing)} numerical features with MEDIAN...")
        num_imputer = SimpleImputer(strategy='median')
        X[numerical_cols_with_missing] = num_imputer.fit_transform(X[numerical_cols_with_missing])
        
        # Save imputer
        with open('models/numerical_imputer.pkl', 'wb') as f:
            pickle.dump(num_imputer, f)
        print("✓ Numerical imputer saved")
    
    # Handle categorical features - use mode (most frequent)
    categorical_cols_with_missing = [col for col in categorical_features if X[col].isnull().any()]
    if categorical_cols_with_missing:
        print(f"\nImputing {len(categorical_cols_with_missing)} categorical features with MODE...")
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols_with_missing] = cat_imputer.fit_transform(X[categorical_cols_with_missing])
        
        # Save imputer
        with open('models/categorical_imputer.pkl', 'wb') as f:
            pickle.dump(cat_imputer, f)
        print("✓ Categorical imputer saved")
    
    # Handle target missing values (if any)
    if y.isnull().any():
        print(f"\n⚠️  Warning: Target has {y.isnull().sum()} missing values")
        # Drop rows with missing target
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"Removed rows with missing target. New shape: {X.shape}")
    
    # Combine back
    df_processed = pd.concat([X, y], axis=1)
    
    # Verify no missing values
    missing_after = df_processed.isnull().sum().sum()
    print(f"\n✓ Missing values after imputation: {missing_after}")
else:
    print("✓ No missing values found!")

# ============================================================================
# 3. REMOVE DUPLICATES
# ============================================================================
print("\n2. Removing Duplicates")
print("-" * 80)

duplicates = df_processed.duplicated().sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate rows")
    df_processed = df_processed.drop_duplicates()
    print(f"✓ Duplicates removed. New shape: {df_processed.shape}")
else:
    print("✓ No duplicates found")

# ============================================================================
# 4. REMOVE ID COLUMNS
# ============================================================================
print("\n3. Removing ID Columns")
print("-" * 80)

# Identify ID columns
id_cols = [col for col in df_processed.columns if 'id' in col.lower() and col.lower() != 'cibil']
if id_cols:
    print(f"Removing ID columns: {id_cols}")
    df_processed = df_processed.drop(columns=id_cols)
    print(f"✓ ID columns removed")
else:
    print("✓ No ID columns found")

# ============================================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================================
print("\n4. Separating Features and Target")
print("-" * 80)

y = df_processed[target_col]
X = df_processed.drop(columns=[target_col])

print(f"Features Shape: {X.shape}")
print(f"Target Shape: {y.shape}")

# Update feature lists (after removing ID columns)
numerical_features = [col for col in X.select_dtypes(include=[np.number]).columns]
categorical_features = [col for col in X.select_dtypes(include=['object']).columns]

print(f"\nUpdated Feature Counts:")
print(f"  • Numerical: {len(numerical_features)}")
print(f"  • Categorical: {len(categorical_features)}")

# ============================================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n5. Encoding Categorical Features")
print("-" * 80)

X_encoded = X.copy()
encoders = {}

if len(categorical_features) > 0:
    for col in categorical_features:
        unique_values = X_encoded[col].nunique()
        
        print(f"\n→ {col} ({unique_values} unique values)")
        
        if unique_values == 2:
            # Binary encoding - Label Encoding
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
            
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"  Label Encoding applied: {mapping}")
            
        elif unique_values <= 10:
            # One-Hot Encoding for low cardinality
            print(f"  One-Hot Encoding applied")
            dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
            X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
            
            encoders[col] = {
                'type': 'onehot',
                'columns': dummies.columns.tolist(),
                'original_values': X[col].unique().tolist()
            }
            print(f"  Created {len(dummies.columns)} dummy variables")
        else:
            # Label Encoding for high cardinality
            print(f"  Label Encoding applied (high cardinality)")
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
    
    print(f"\n✓ All categorical features encoded")
else:
    print("No categorical features to encode")

print(f"Final feature count: {X_encoded.shape[1]}")

# Save encoders
with open('models/feature_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("✓ Feature encoders saved")

# ============================================================================
# 7. ENCODE TARGET VARIABLE
# ============================================================================
print("\n6. Encoding Target Variable")
print("-" * 80)

# Encode target to 0 and 1
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y.astype(str))

print(f"Target classes: {target_encoder.classes_}")
print(f"Encoded mapping: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
print(f"\nEncoded target distribution:")
unique, counts = np.unique(y_encoded, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  Class {val}: {count} ({count/len(y_encoded)*100:.2f}%)")

# Save target encoder
with open('models/target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
print("\n✓ Target encoder saved")

# ============================================================================
# 8. FEATURE SCALING
# ============================================================================
print("\n7. Feature Scaling")
print("-" * 80)

print("Applying StandardScaler to numerical features...")
print("(Centers features to mean=0, std=1)")

# Scale all features (numerical are already numeric, encoded categoricals are now numeric too)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_encoded),
    columns=X_encoded.columns,
    index=X_encoded.index
)

print("\nScaling Statistics (first 5 features):")
print(X_scaled.iloc[:, :5].describe().loc[['mean', 'std']].round(3))

# Save scaler
with open('models/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\n✓ Feature scaler saved")

# ============================================================================
# 9. DATA QUALITY VERIFICATION
# ============================================================================
print("\n8. Data Quality Verification")
print("-" * 80)

issues = []

# Check for missing values
if X_scaled.isnull().sum().sum() > 0:
    issues.append("Missing values found after preprocessing")

# Check for infinite values
if np.isinf(X_scaled.values).sum() > 0:
    issues.append("Infinite values found after preprocessing")

# Check for constant features
constant_features = X_scaled.columns[X_scaled.nunique() == 1].tolist()
if constant_features:
    issues.append(f"Constant features found: {constant_features}")
    # Remove constant features
    X_scaled = X_scaled.drop(columns=constant_features)
    X_encoded = X_encoded.drop(columns=constant_features)
    print(f"  Removed {len(constant_features)} constant features")

if len(issues) > 0:
    print("Issues detected:")
    for issue in issues:
        print(f"  • {issue}")
else:
    print("✓ All quality checks passed!")
    print("  • No missing values")
    print("  • No infinite values")
    print("  • All features have variation")

# ============================================================================
# 10. SAVE PREPROCESSED DATA
# ============================================================================
print("\n9. Saving Preprocessed Data")
print("-" * 80)

# Save unscaled encoded features (for tree-based models)
X_encoded.to_csv('data/X_encoded.csv', index=False)
print("✓ Saved: data/X_encoded.csv (unscaled, encoded features)")

# Save scaled features (for logistic regression)
X_scaled.to_csv('data/X_scaled.csv', index=False)
print("✓ Saved: data/X_scaled.csv (scaled features)")

# Save encoded target
pd.Series(y_encoded, name=target_col).to_csv('data/y_encoded.csv', index=False)
print("✓ Saved: data/y_encoded.csv (encoded target)")

# Save feature names
feature_info = {
    'all_features': X_scaled.columns.tolist(),
    'original_numerical': numerical_features,
    'original_categorical': categorical_features,
    'target_column': target_col,
    'target_classes': target_encoder.classes_.tolist()
}

with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=4)
print("✓ Saved: models/feature_info.json")

# ============================================================================
# 11. PREPROCESSING SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)

print(f"""
Original Dataset:
  • Records: {len(df)}
  • Features: {df.shape[1]}
  • Missing Values: {summary['missing_values']}
  • Duplicates: {summary['duplicates']}

Processed Dataset:
  • Records: {len(X_scaled)}
  • Features: {X_scaled.shape[1]}
  • Missing Values: {X_scaled.isnull().sum().sum()}
  • All Numerical: Yes

Preprocessing Steps Applied:
  ✓ Missing values imputed (median for numeric, mode for categorical)
  ✓ Duplicates removed ({duplicates} rows)
  ✓ ID columns removed
  ✓ Categorical features encoded ({len(categorical_features)} features)
  ✓ Features scaled (StandardScaler)
  ✓ Target encoded to binary (0/1)

Target Variable:
  • Original: {target_col}
  • Classes: {target_encoder.classes_.tolist()}
  • Distribution: Class 0: {counts[0]} ({counts[0]/len(y_encoded)*100:.1f}%), Class 1: {counts[1]} ({counts[1]/len(y_encoded)*100:.1f}%)
  • Imbalance Ratio: {max(counts)/min(counts):.2f}:1

Saved Artifacts:
  ✓ Encoded features (scaled and unscaled)
  ✓ Encoded target
  ✓ Feature encoders
  ✓ Target encoder
  ✓ Feature scaler
  ✓ Imputers (if needed)
  ✓ Feature information

Data Ready For:
  ✓ Train-test split
  ✓ SMOTE application (class imbalance handling)
  ✓ Model training (Logistic Regression & Decision Tree)

Next Steps:
  1. Split data into train/test sets (3_train_test_split.py)
  2. Apply SMOTE to training data
  3. Train classification models
  4. Evaluate with precision, recall, F1-score
""")

print("="*80)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*80)