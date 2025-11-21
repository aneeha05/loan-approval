"""
Loan Approval Prediction - Step 5: Decision Tree Model
Purpose: Train and evaluate Decision Tree classifier
Focus: Precision, Recall, F1-Score on imbalanced data
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================================
print("="*80)
print("LOAN APPROVAL PREDICTION - DECISION TREE MODEL")
print("="*80)

# Load unscaled data (Decision Trees don't require scaling)
X_train = pd.read_csv('data/X_train_unscaled.csv')
X_test = pd.read_csv('data/X_test_unscaled.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Load split info
with open('data/split_info.json', 'r') as f:
    split_info = json.load(f)

# Load feature info
with open('models/feature_info.json', 'r') as f:
    feature_info = json.load(f)

target_col = feature_info['target_column']

print(f"\nLoaded Data:")
print(f"  • Training Set: {X_train.shape}")
print(f"  • Test Set: {X_test.shape}")
print(f"  • SMOTE Applied: {split_info['smote_applied']}")
print(f"  • Training Distribution: {split_info['train_class_distribution']}")

# ============================================================================
# 2. TRAIN DECISION TREE MODEL
# ============================================================================
print("\n1. Training Decision Tree Model")
print("-" * 80)

# Initialize model with parameters to prevent overfitting
model_dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,                    # Limit tree depth
    min_samples_split=20,            # Minimum samples to split
    min_samples_leaf=10,             # Minimum samples in leaf
    class_weight='balanced',         # Handle any remaining imbalance
    criterion='gini'                 # Gini impurity
)

print("Model Configuration:")
print(f"  • Criterion: gini")
print(f"  • Max Depth: 10")
print(f"  • Min Samples Split: 20")
print(f"  • Min Samples Leaf: 10")
print(f"  • Class Weight: balanced")

print("\nTraining model...")
model_dt.fit(X_train, y_train)
print("✓ Model training completed!")

# Get tree information
print(f"\nTree Structure:")
print(f"  • Number of Leaves: {model_dt.get_n_leaves()}")
print(f"  • Max Depth Reached: {model_dt.get_depth()}")
print(f"  • Total Nodes: {model_dt.tree_.node_count}")

# ============================================================================
# 3. MAKE PREDICTIONS
# ============================================================================
print("\n2. Making Predictions")
print("-" * 80)

# Predictions on training set
y_train_pred = model_dt.predict(X_train)
y_train_proba = model_dt.predict_proba(X_train)[:, 1]

# Predictions on test set
y_test_pred = model_dt.predict(X_test)
y_test_proba = model_dt.predict_proba(X_test)[:, 1]

print(f"Training predictions: {len(y_train_pred)} samples")
print(f"Test predictions: {len(y_test_pred)} samples")

# Sample predictions
print("\nSample Test Predictions:")
sample_df = pd.DataFrame({
    'Actual': y_test[:5],
    'Predicted': y_test_pred[:5],
    'Probability': y_test_proba[:5]
})
print(sample_df.to_string(index=False))

# ============================================================================
# 4. EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n3. Model Performance Evaluation")
print("-" * 80)

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='binary')
train_recall = recall_score(y_train, y_train_pred, average='binary')
train_f1 = f1_score(y_train, y_train_pred, average='binary')

print("TRAINING SET PERFORMANCE:")
print(f"  • Accuracy:  {train_accuracy:.4f}")
print(f"  • Precision: {train_precision:.4f}")
print(f"  • Recall:    {train_recall:.4f}")
print(f"  • F1-Score:  {train_f1:.4f}")

# Test metrics (PRIMARY EVALUATION)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='binary')
test_recall = recall_score(y_test, y_test_pred, average='binary')
test_f1 = f1_score(y_test, y_test_pred, average='binary')
test_roc_auc = roc_auc_score(y_test, y_test_proba)

print("\nTEST SET PERFORMANCE (PRIMARY METRICS):")
print(f"  • Accuracy:  {test_accuracy:.4f}")
print(f"  • Precision: {test_precision:.4f} ⭐ (% of predicted approvals that are correct)")
print(f"  • Recall:    {test_recall:.4f} ⭐ (% of actual approvals we catch)")
print(f"  • F1-Score:  {test_f1:.4f} ⭐ (Harmonic mean of precision & recall)")
print(f"  • ROC-AUC:   {test_roc_auc:.4f}")

# Check for overfitting
print("\n📊 OVERFITTING CHECK:")
accuracy_diff = train_accuracy - test_accuracy
if accuracy_diff > 0.1:
    print(f"  ⚠️  Possible overfitting detected!")
    print(f"  Training accuracy ({train_accuracy:.3f}) >> Test accuracy ({test_accuracy:.3f})")
else:
    print(f"  ✓ Good generalization")
    print(f"  Accuracy difference: {accuracy_diff:.3f}")

# Interpretation
print("\n📊 METRIC INTERPRETATION:")
print(f"  • Precision {test_precision:.2%}: Of all loans we approve, {test_precision:.1%} should actually be approved")
print(f"  • Recall {test_recall:.2%}: We catch {test_recall:.1%} of all loans that should be approved")
print(f"  • F1-Score {test_f1:.2%}: Overall balanced performance measure")

# ============================================================================
# 5. DETAILED CLASSIFICATION REPORT
# ============================================================================
print("\n4. Detailed Classification Report")
print("-" * 80)

print("\nTRAINING SET:")
print(classification_report(y_train, y_train_pred, 
                           target_names=['Rejected', 'Approved'],
                           digits=4))

print("\nTEST SET:")
print(classification_report(y_test, y_test_pred, 
                           target_names=['Rejected', 'Approved'],
                           digits=4))

# ============================================================================
# 6. CONFUSION MATRIX
# ============================================================================
print("\n5. Confusion Matrix Analysis")
print("-" * 80)

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (Test Set):")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nBreakdown:")
print(f"  • True Negatives (TN):  {tn} (Correctly rejected)")
print(f"  • False Positives (FP): {fp} (Wrongly approved - COSTLY)")
print(f"  • False Negatives (FN): {fn} (Wrongly rejected - Lost opportunity)")
print(f"  • True Positives (TP):  {tp} (Correctly approved)")

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Test set confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0],
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Add percentages
for i in range(2):
    for j in range(2):
        text = axes[0].text(j + 0.5, i + 0.7,
                          f'({cm[i, j]/cm.sum()*100:.1f}%)',
                          ha="center", va="center", color="red", fontsize=9)

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges', ax=axes[1],
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'],
            cbar_kws={'label': 'Percentage'})
axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('results/decision_tree_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix plot saved")

# ============================================================================
# 7. ROC CURVE AND PRECISION-RECALL CURVE
# ============================================================================
print("\n6. ROC and Precision-Recall Curves")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_test_proba)
axes[0].plot(fpr, tpr, color='purple', lw=2, 
            label=f'ROC curve (AUC = {test_roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve - Decision Tree', fontsize=14, fontweight='bold')
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# Precision-Recall Curve
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_test_proba)
avg_precision = average_precision_score(y_test, y_test_proba)

axes[1].plot(recall_curve, precision_curve, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/decision_tree_roc_pr_curves.png', dpi=300, bbox_inches='tight')
print("✓ ROC and PR curves saved")

# ============================================================================
# 8. FEATURE IMPORTANCE (Gini Importance)
# ============================================================================
print("\n7. Feature Importance Analysis")
print("-" * 80)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model_dt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize top features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], color='coral', alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance (Gini)', fontsize=12)
plt.title('Top 15 Feature Importances - Decision Tree', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance plot saved")

# Save feature importance
feature_importance.to_csv('results/decision_tree_feature_importance.csv', index=False)

# ============================================================================
# 9. VISUALIZE DECISION TREE (Top Levels)
# ============================================================================
print("\n8. Visualizing Decision Tree Structure")
print("-" * 80)

# Plot top 3 levels of the tree
plt.figure(figsize=(20, 10))
plot_tree(model_dt, 
          max_depth=3,
          feature_names=X_train.columns,
          class_names=['Rejected', 'Approved'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Structure (Top 3 Levels)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/decision_tree_structure.png', dpi=300, bbox_inches='tight')
print("✓ Decision tree structure plot saved")

# ============================================================================
# 10. SAVE MODEL AND RESULTS
# ============================================================================
print("\n9. Saving Model and Results")
print("-" * 80)

# Save trained model
with open('models/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(model_dt, f)
print("✓ Saved: models/decision_tree_model.pkl")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Probability': y_test_proba,
    'Correct': y_test == y_test_pred
})
predictions_df.to_csv('results/decision_tree_predictions.csv', index=False)
print("✓ Saved: results/decision_tree_predictions.csv")

# Save performance metrics
performance_metrics = {
    'model_name': 'Decision Tree',
    'smote_applied': split_info['smote_applied'],
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'n_features': X_train.shape[1],
    'tree_depth': int(model_dt.get_depth()),
    'n_leaves': int(model_dt.get_n_leaves()),
    'train_metrics': {
        'accuracy': float(train_accuracy),
        'precision': float(train_precision),
        'recall': float(train_recall),
        'f1_score': float(train_f1)
    },
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'roc_auc': float(test_roc_auc),
        'average_precision': float(avg_precision)
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
}

with open('results/decision_tree_performance.json', 'w') as f:
    json.dump(performance_metrics, f, indent=4)
print("✓ Saved: results/decision_tree_performance.json")

# ============================================================================
# 11. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("DECISION TREE MODEL SUMMARY")
print("="*80)

print(f"""
Model: Decision Tree Classifier
Target: {target_col}
Training Samples: {len(X_train)} (SMOTE balanced)
Test Samples: {len(X_test)} (original distribution)
Features: {X_train.shape[1]}
Tree Depth: {model_dt.get_depth()}
Number of Leaves: {model_dt.get_n_leaves()}

📊 TEST SET PERFORMANCE (Key Metrics):

  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
    → Overall correctness of predictions
  
  Precision: {test_precision:.4f} ({test_precision*100:.2f}%) ⭐
    → Of predicted approvals, {test_precision*100:.1f}% are correct
    → HIGH precision = Few bad loans approved
    → CRITICAL for risk management
  
  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%) ⭐
    → Model catches {test_recall*100:.1f}% of all good loans
    → HIGH recall = Don't miss good opportunities
    → CRITICAL for business growth
  
  F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%) ⭐
    → Balanced measure of precision & recall
    → Best for imbalanced data
  
  ROC-AUC:   {test_roc_auc:.4f}
    → Model's ability to distinguish classes

Confusion Matrix Breakdown:
  • True Negatives:  {tn} (✓ Correctly rejected bad loans)
  • False Positives: {fp} (✗ Wrongly approved - RISK!)
  • False Negatives: {fn} (✗ Wrongly rejected - Lost opportunity)
  • True Positives:  {tp} (✓ Correctly approved good loans)

Business Impact:
  • False Positive Rate: {fp/(fp+tn)*100:.2f}% (Type I Error)
    → Risk: Approving bad loans
  
  • False Negative Rate: {fn/(fn+tp)*100:.2f}% (Type II Error)
    → Risk: Rejecting good applicants

Overfitting Check:
  • Training Accuracy: {train_accuracy:.4f}
  • Test Accuracy: {test_accuracy:.4f}
  • Difference: {train_accuracy - test_accuracy:.4f}
  • Status: {'⚠️ Possible overfitting' if train_accuracy - test_accuracy > 0.1 else '✓ Good generalization'}

Model Characteristics:
  ✓ Non-parametric (no assumptions about data distribution)
  ✓ Handles non-linear relationships naturally
  ✓ Feature importance based on information gain
  ✓ Interpretable through tree visualization
  ✓ No feature scaling required

Top 3 Most Important Features:
{chr(10).join([f'  {i+1}. {row["Feature"]}: {row["Importance"]:.4f}' 
               for i, row in feature_importance.head(3).iterrows()])}

Saved Artifacts:
  ✓ Trained model
  ✓ Predictions with probabilities
  ✓ Performance metrics
  ✓ Feature importance
  ✓ Tree structure visualization
  ✓ Confusion matrix, ROC, PR curves

Next Steps:
  1. Compare with Logistic Regression (6_model_comparison.py)
  2. Select best model for deployment
  3. Consider ensemble methods if needed
  4. Tune decision threshold for business requirements
""")

print("="*80)
print("DECISION TREE MODEL COMPLETED!")
print("="*80)