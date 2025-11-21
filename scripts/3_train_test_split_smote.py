"""
Loan Approval Prediction - Step 3: Train-Test Split and SMOTE
Purpose: Split data and handle class imbalance using SMOTE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(i) for i in obj]
    else:
        return obj
# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("="*80)
print("LOAN APPROVAL PREDICTION - TRAIN-TEST SPLIT & SMOTE")
print("="*80)

# Load encoded features (scaled for logistic regression)
X_scaled = pd.read_csv('data/X_scaled.csv')
# Load encoded features (unscaled for decision tree)
X_encoded = pd.read_csv('data/X_encoded.csv')
# Load encoded target
y = pd.read_csv('data/y_encoded.csv').values.ravel()

# Load feature info
with open('models/feature_info.json', 'r') as f:
    feature_info = json.load(f)

target_col = feature_info['target_column']

print(f"\nLoaded Data:")
print(f"  • Features (Scaled): {X_scaled.shape}")
print(f"  • Features (Unscaled): {X_encoded.shape}")
print(f"  • Target: {y.shape}")

# ============================================================================
# 2. ANALYZE CLASS DISTRIBUTION
# ============================================================================
print("\n1. Class Distribution Analysis")
print("-" * 80)

class_counts = Counter(y)
print(f"Original Class Distribution:")
for cls, count in sorted(class_counts.items()):
    print(f"  • Class {cls}: {count} samples ({count/len(y)*100:.2f}%)")

# Calculate imbalance ratio
majority_class = max(class_counts.values())
minority_class = min(class_counts.values())
imbalance_ratio = majority_class / minority_class

print(f"\nImbalance Analysis:")
print(f"  • Majority Class: {majority_class} samples")
print(f"  • Minority Class: {minority_class} samples")
print(f"  • Imbalance Ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print(f"  • Status: ⚠️  IMBALANCED - SMOTE will be applied")
    apply_smote = True
else:
    print(f"  • Status: ✓ Balanced - SMOTE not needed")
    apply_smote = False

# Visualize original distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts, color=['#e74c3c', '#2ecc71'], alpha=0.8)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Original Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(unique)
for i, (cls, cnt) in enumerate(zip(unique, counts)):
    plt.text(cls, cnt, f'{cnt}\n({cnt/len(y)*100:.1f}%)', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# 3. STRATIFIED TRAIN-TEST SPLIT
# ============================================================================
print("\n2. Performing Stratified Train-Test Split")
print("-" * 80)

test_size = 0.2
random_state = 42

# Split scaled data (for Logistic Regression)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# Split unscaled data (for Decision Tree)
X_train_unscaled, X_test_unscaled, _, _ = train_test_split(
    X_encoded, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

print(f"Split Configuration:")
print(f"  • Test Size: {test_size*100:.0f}%")
print(f"  • Random State: {random_state}")
print(f"  • Stratified: Yes (maintains class distribution)")

print(f"\nResulting Splits:")
print(f"  • Training Set: {len(X_train_scaled)} samples ({len(X_train_scaled)/len(X_scaled)*100:.1f}%)")
print(f"  • Test Set: {len(X_test_scaled)} samples ({len(X_test_scaled)/len(X_scaled)*100:.1f}%)")

# Check class distribution in splits
train_dist = Counter(y_train)
test_dist = Counter(y_test)

print(f"\nTraining Set Distribution:")
for cls, count in sorted(train_dist.items()):
    print(f"  • Class {cls}: {count} ({count/len(y_train)*100:.2f}%)")

print(f"\nTest Set Distribution:")
for cls, count in sorted(test_dist.items()):
    print(f"  • Class {cls}: {count} ({count/len(y_test)*100:.2f}%)")

# ============================================================================
# 4. APPLY SMOTE TO TRAINING DATA
# ============================================================================
if apply_smote:
    print("\n3. Applying SMOTE (Synthetic Minority Over-sampling Technique)")
    print("-" * 80)
    
    print("SMOTE will create synthetic samples for the minority class")
    print("to balance the training data.\n")
    
    print("Before SMOTE:")
    print(f"  • Class Distribution: {dict(train_dist)}")
    print(f"  • Training Samples: {len(X_train_scaled)}")
    
    # Apply SMOTE to scaled data (for Logistic Regression)
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_scaled_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Apply SMOTE to unscaled data (for Decision Tree)
    X_train_unscaled_balanced, _ = smote.fit_resample(X_train_unscaled, y_train)
    
    balanced_dist = Counter(y_train_balanced)
    
    print(f"\nAfter SMOTE:")
    print(f"  • Class Distribution: {dict(balanced_dist)}")
    print(f"  • Training Samples: {len(X_train_scaled_balanced)}")
    print(f"  • Synthetic Samples Created: {len(X_train_scaled_balanced) - len(X_train_scaled)}")
    
    new_ratio = max(balanced_dist.values()) / min(balanced_dist.values())
    print(f"  • New Imbalance Ratio: {new_ratio:.2f}:1")
    print(f"\n✓ SMOTE applied successfully!")
    
    # Visualize after SMOTE
    plt.subplot(1, 2, 2)
    unique_bal, counts_bal = np.unique(y_train_balanced, return_counts=True)
    plt.bar(unique_bal, counts_bal, color=['#e74c3c', '#2ecc71'], alpha=0.8)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('After SMOTE (Training Set)', fontsize=14, fontweight='bold')
    plt.xticks(unique_bal)
    for i, (cls, cnt) in enumerate(zip(unique_bal, counts_bal)):
        plt.text(cls, cnt, f'{cnt}\n({cnt/len(y_train_balanced)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/07_smote_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ SMOTE comparison plot saved")
    
    # Convert back to DataFrame
    X_train_scaled_balanced = pd.DataFrame(X_train_scaled_balanced, columns=X_scaled.columns)
    X_train_unscaled_balanced = pd.DataFrame(X_train_unscaled_balanced, columns=X_encoded.columns)
    
else:
    print("\n3. Class Imbalance Handling")
    print("-" * 80)
    print("✓ Data is relatively balanced, SMOTE not applied")
    X_train_scaled_balanced = X_train_scaled
    X_train_unscaled_balanced = X_train_unscaled
    y_train_balanced = y_train

# ============================================================================
# 5. SAVE TRAIN-TEST SPLITS
# ============================================================================
print("\n4. Saving Train-Test Splits")
print("-" * 80)

# Save scaled splits (for Logistic Regression)
X_train_scaled_balanced.to_csv('data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/X_test_scaled.csv', index=False)
print("✓ Saved: Scaled features (for Logistic Regression)")

# Save unscaled splits (for Decision Tree)
X_train_unscaled_balanced.to_csv('data/X_train_unscaled.csv', index=False)
X_test_unscaled.to_csv('data/X_test_unscaled.csv', index=False)
print("✓ Saved: Unscaled features (for Decision Tree)")

# Save target splits
pd.Series(y_train_balanced, name=target_col).to_csv('data/y_train.csv', index=False)
pd.Series(y_test, name=target_col).to_csv('data/y_test.csv', index=False)
print("✓ Saved: Target variables")

# Also save original unbalanced training data for comparison
pd.Series(y_train, name=target_col).to_csv('data/y_train_original.csv', index=False)
print("✓ Saved: Original training target (before SMOTE)")

# ============================================================================
# 6. SAVE SPLIT INFORMATION
# ============================================================================
print("\n5. Saving Split Information")
print("-" * 80)

split_info = {
    'test_size': test_size,
    'random_state': random_state,
    'stratified': True,
    'smote_applied': apply_smote,
    'original_train_samples': len(X_train_scaled),
    'balanced_train_samples': len(X_train_scaled_balanced),
    'test_samples': len(X_test_scaled),
    'synthetic_samples_created': len(X_train_scaled_balanced) - len(X_train_scaled) if apply_smote else 0,
    'train_class_distribution': dict(Counter(y_train_balanced)),
    'test_class_distribution': dict(test_dist),
    'original_imbalance_ratio': float(imbalance_ratio),
    'balanced_imbalance_ratio': float(max(Counter(y_train_balanced).values()) / min(Counter(y_train_balanced).values()))
}

clean_split_info = convert_keys_to_str(split_info)

with open("data/split_info.json", "w") as f:
    json.dump(clean_split_info, f, indent=4)
# ============================================================================
# 7. DATA VERIFICATION
# ============================================================================
print("\n6. Data Verification")
print("-" * 80)

checks = []

# Check 1: No missing values
if (X_train_scaled_balanced.isnull().sum().sum() == 0 and 
    X_test_scaled.isnull().sum().sum() == 0):
    checks.append(("No missing values", True))
else:
    checks.append(("No missing values", False))

# Check 2: Same features in train and test
if (list(X_train_scaled_balanced.columns) == list(X_test_scaled.columns)):
    checks.append(("Feature alignment", True))
else:
    checks.append(("Feature alignment", False))

# Check 3: Adequate sample sizes
if len(X_train_scaled_balanced) > 100 and len(X_test_scaled) > 20:
    checks.append(("Adequate sample sizes", True))
else:
    checks.append(("Adequate sample sizes", False))

# Check 4: Both classes present in splits
if (len(np.unique(y_train_balanced)) == 2 and len(np.unique(y_test)) == 2):
    checks.append(("Both classes present", True))
else:
    checks.append(("Both classes present", False))

# Check 5: Test set unchanged (no SMOTE on test)
if len(y_test) == int(len(y) * test_size):
    checks.append(("Test set preserved", True))
else:
    checks.append(("Test set preserved", False))

print("Verification Checklist:")
for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"  {status} {check_name}")

all_passed = all(check[1] for check in checks)
if all_passed:
    print("\n✅ All checks passed! Data ready for model training.")
else:
    print("\n⚠️  Some checks failed. Review before proceeding.")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("TRAIN-TEST SPLIT & SMOTE SUMMARY")
print("="*80)

print(f"""
Dataset Split:
  • Original Records: {len(X_scaled)}
  • Test Size: {test_size*100:.0f}%
  • Stratified: Yes
  
Training Set:
  • Original Samples: {len(X_train_scaled)}
  • After SMOTE: {len(X_train_scaled_balanced)}
  • Synthetic Samples Created: {len(X_train_scaled_balanced) - len(X_train_scaled)}
  • Class Distribution: {dict(Counter(y_train_balanced))}
  • Imbalance Ratio: {max(Counter(y_train_balanced).values()) / min(Counter(y_train_balanced).values()):.2f}:1

Test Set (Unchanged):
  • Samples: {len(X_test_scaled)}
  • Class Distribution: {dict(test_dist)}
  • Imbalance Ratio: {max(test_dist.values()) / min(test_dist.values()):.2f}:1

SMOTE Application:
  • Applied: {'Yes' if apply_smote else 'No'}
  • Reason: {'Class imbalance detected' if apply_smote else 'Data is balanced'}
  • Algorithm: SMOTE with k_neighbors=5
  • Effect: {'Balanced training set to 50-50' if apply_smote else 'N/A'}

Saved Files:
  ✓ X_train_scaled.csv (for Logistic Regression)
  ✓ X_train_unscaled.csv (for Decision Tree)
  ✓ X_test_scaled.csv
  ✓ X_test_unscaled.csv
  ✓ y_train.csv (balanced)
  ✓ y_test.csv
  ✓ split_info.json

Why SMOTE on Training Only?
  • Test set represents real-world distribution
  • SMOTE only helps model learn from balanced data
  • We evaluate on original imbalanced test distribution
  • This prevents data leakage and overoptimistic results

Data Ready For:
  ✓ Logistic Regression training
  ✓ Decision Tree training
  ✓ Model comparison
  ✓ Evaluation with precision, recall, F1-score

Next Steps:
  1. Train Logistic Regression model (4_logistic_regression.py)
  2. Train Decision Tree model (5_decision_tree.py)
  3. Compare model performance
  4. Focus on precision, recall, F1-score metrics
""")

print("="*80)
print("TRAIN-TEST SPLIT & SMOTE COMPLETED SUCCESSFULLY!")
print("="*80)