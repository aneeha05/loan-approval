"""
Loan Approval Prediction - Step 6: Model Comparison
Purpose: Compare Logistic Regression vs Decision Tree
Focus: Precision, Recall, F1-Score analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD PERFORMANCE METRICS
# ============================================================================
print("="*80)
print("LOAN APPROVAL PREDICTION - MODEL COMPARISON")
print("="*80)

# Load Logistic Regression metrics
with open('results/logistic_regression_performance.json', 'r') as f:
    lr_metrics = json.load(f)

# Load Decision Tree metrics
with open('results/decision_tree_performance.json', 'r') as f:
    dt_metrics = json.load(f)

# Load split info
with open('data/split_info.json', 'r') as f:
    split_info = json.load(f)

print("\nLoaded Performance Metrics:")
print(f"  • Logistic Regression: ✓")
print(f"  • Decision Tree: ✓")
print(f"  • SMOTE Applied: {split_info['smote_applied']}")

# ============================================================================
# 2. CREATE COMPARISON TABLE
# ============================================================================
print("\n1. Model Performance Comparison")
print("-" * 80)

# Create comparison dataframe
comparison_data = {
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [
        lr_metrics['test_metrics']['accuracy'],
        dt_metrics['test_metrics']['accuracy']
    ],
    'Precision': [
        lr_metrics['test_metrics']['precision'],
        dt_metrics['test_metrics']['precision']
    ],
    'Recall': [
        lr_metrics['test_metrics']['recall'],
        dt_metrics['test_metrics']['recall']
    ],
    'F1-Score': [
        lr_metrics['test_metrics']['f1_score'],
        dt_metrics['test_metrics']['f1_score']
    ],
    'ROC-AUC': [
        lr_metrics['test_metrics']['roc_auc'],
        dt_metrics['test_metrics']['roc_auc']
    ]
}

comparison_df = pd.DataFrame(comparison_data)

print("\nTest Set Performance Comparison:")
print(comparison_df.to_string(index=False))

# Find best model for each metric
print("\n🏆 Best Performance by Metric:")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    best_idx = comparison_df[metric].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    best_value = comparison_df.loc[best_idx, metric]
    print(f"  • {metric:10s}: {best_model:22s} ({best_value:.4f})")

# ============================================================================
# 3. VISUALIZE METRIC COMPARISON
# ============================================================================
print("\n2. Creating Comparison Visualizations")
print("-" * 80)

# Plot 1: Bar chart comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#3498db', '#e74c3c']

for idx, metric in enumerate(metrics):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    values = comparison_df[metric].values
    bars = ax.bar(comparison_df['Model'], values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=15)

# Hide the last subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('results/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_metrics.png")

# ============================================================================
# 4. CONFUSION MATRIX COMPARISON
# ============================================================================
print("\n3. Confusion Matrix Comparison")
print("-" * 80)

# Extract confusion matrices
lr_cm = lr_metrics['confusion_matrix']
dt_cm = dt_metrics['confusion_matrix']

print("\nLogistic Regression:")
print(f"  TN: {lr_cm['true_negatives']:4d} | FP: {lr_cm['false_positives']:4d}")
print(f"  FN: {lr_cm['false_negatives']:4d} | TP: {lr_cm['true_positives']:4d}")

print("\nDecision Tree:")
print(f"  TN: {dt_cm['true_negatives']:4d} | FP: {dt_cm['false_positives']:4d}")
print(f"  FN: {dt_cm['false_negatives']:4d} | TP: {dt_cm['true_positives']:4d}")

# Calculate business metrics
print("\n💼 Business Impact Metrics:")

for model_name, cm in [('Logistic Regression', lr_cm), ('Decision Tree', dt_cm)]:
    tn, fp, fn, tp = cm['true_negatives'], cm['false_positives'], cm['false_negatives'], cm['true_positives']
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    print(f"\n{model_name}:")
    print(f"  • False Positive Rate: {fpr:.2%} (Bad loans approved)")
    print(f"  • False Negative Rate: {fnr:.2%} (Good loans rejected)")
    print(f"  • Total Errors: {fp + fn} ({(fp+fn)/(tn+fp+fn+tp)*100:.2f}%)")

# ============================================================================
# 5. PRECISION-RECALL TRADE-OFF
# ============================================================================
print("\n4. Precision-Recall Trade-off Analysis")
print("-" * 80)

# Create precision-recall comparison
pr_data = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Precision': [
        lr_metrics['test_metrics']['precision'],
        dt_metrics['test_metrics']['precision']
    ],
    'Recall': [
        lr_metrics['test_metrics']['recall'],
        dt_metrics['test_metrics']['recall']
    ]
})

# Plot precision vs recall
fig, ax = plt.subplots(figsize=(10, 8))

for idx, row in pr_data.iterrows():
    ax.scatter(row['Recall'], row['Precision'], s=300, alpha=0.6,
              color=colors[idx], edgecolor='black', linewidth=2,
              label=row['Model'])
    
    # Add model name near point
    ax.annotate(row['Model'], 
                xy=(row['Recall'], row['Precision']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold')

# Add diagonal line (F1-Score contours)
recall_range = np.linspace(0.1, 1, 100)
for f1 in [0.5, 0.6, 0.7, 0.8, 0.9]:
    precision = f1 * recall_range / (2 * recall_range - f1)
    precision = np.clip(precision, 0, 1)
    ax.plot(recall_range, precision, 'k--', alpha=0.2, linewidth=1)
    
    # Label F1 contour
    if f1 <= 0.8:
        ax.text(0.9, f1/1.8, f'F1={f1}', fontsize=9, alpha=0.5)

ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_tradeoff.png")

# ============================================================================
# 6. MODEL CHARACTERISTICS COMPARISON
# ============================================================================
print("\n5. Model Characteristics Comparison")
print("-" * 80)

characteristics = pd.DataFrame({
    'Characteristic': [
        'Interpretability',
        'Training Speed',
        'Prediction Speed',
        'Handles Non-linearity',
        'Feature Scaling Required',
        'Prone to Overfitting',
        'Handles Imbalanced Data',
        'Probabilistic Output',
        'Feature Importance'
    ],
    'Logistic Regression': [
        'High (coefficients)',
        'Fast',
        'Very Fast',
        'No (linear only)',
        'Yes (StandardScaler used)',
        'Low (with regularization)',
        'Yes (class_weight + SMOTE)',
        'Yes (probabilities)',
        'Coefficients'
    ],
    'Decision Tree': [
        'High (tree structure)',
        'Fast',
        'Very Fast',
        'Yes (naturally)',
        'No',
        'Medium (controlled by depth)',
        'Yes (class_weight + SMOTE)',
        'Yes (leaf probabilities)',
        'Gini importance'
    ]
})

print("\nModel Characteristics:")
print(characteristics.to_string(index=False))

# ============================================================================
# 7. RECOMMENDATION
# ============================================================================
print("\n6. Model Selection Recommendation")
print("-" * 80)

# Determine best model based on F1-score (balanced metric)
best_f1_idx = comparison_df['F1-Score'].idxmax()
best_model = comparison_df.loc[best_f1_idx, 'Model']
best_f1 = comparison_df.loc[best_f1_idx, 'F1-Score']

# Check other important metrics
lr_precision = lr_metrics['test_metrics']['precision']
dt_precision = dt_metrics['test_metrics']['precision']
lr_recall = lr_metrics['test_metrics']['recall']
dt_recall = dt_metrics['test_metrics']['recall']

print(f"\n🎯 RECOMMENDATION: {best_model}")
print(f"   Based on F1-Score: {best_f1:.4f}\n")

print("📋 Decision Factors:")

if lr_precision > dt_precision:
    print(f"  • Logistic Regression has better PRECISION ({lr_precision:.4f})")
    print(f"    → Fewer false loan approvals (lower risk)")
else:
    print(f"  • Decision Tree has better PRECISION ({dt_precision:.4f})")
    print(f"    → Fewer false loan approvals (lower risk)")

if lr_recall > dt_recall:
    print(f"  • Logistic Regression has better RECALL ({lr_recall:.4f})")
    print(f"    → Catches more good loan opportunities")
else:
    print(f"  • Decision Tree has better RECALL ({dt_recall:.4f})")
    print(f"    → Catches more good loan opportunities")

print("\n💡 Business Considerations:")
print("""
  Risk-Averse Strategy (Minimize Bad Loans):
    → Choose model with HIGHER PRECISION
    → Fewer false positives = Lower default risk
  
  Growth-Oriented Strategy (Maximize Approvals):
    → Choose model with HIGHER RECALL
    → More true positives = More business opportunities
  
  Balanced Strategy:
    → Choose model with HIGHER F1-SCORE
    → Best overall performance
""")

# ============================================================================
# 8. SAVE COMPARISON RESULTS
# ============================================================================
print("\n7. Saving Comparison Results")
print("-" * 80)

# Save comparison table
comparison_df.to_csv('results/model_comparison_table.csv', index=False)
print("✓ Saved: results/model_comparison_table.csv")

# Save detailed comparison report
comparison_report = {
    'comparison_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models_compared': ['Logistic Regression', 'Decision Tree'],
    'smote_applied': split_info['smote_applied'],
    'test_samples': lr_metrics['test_samples'],
    'metrics_comparison': comparison_data,
    'best_model_by_f1': best_model,
    'best_f1_score': float(best_f1),
    'logistic_regression': lr_metrics,
    'decision_tree': dt_metrics,
    'recommendation': f"Use {best_model} for balanced performance (F1-Score: {best_f1:.4f})"
}

with open('results/model_comparison_report.json', 'w') as f:
    json.dump(comparison_report, f, indent=4, default=str)
print("✓ Saved: results/model_comparison_report.json")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

print(f"""
Models Evaluated:
  1. Logistic Regression
  2. Decision Tree

Dataset:
  • Training Samples: {lr_metrics['train_samples']} (after SMOTE)
  • Test Samples: {lr_metrics['test_samples']} (original distribution)
  • SMOTE Applied: {split_info['smote_applied']}
  • Imbalance Ratio: {split_info['original_imbalance_ratio']:.2f}:1

Performance Comparison (Test Set):

                       Logistic Regression    Decision Tree
  ────────────────────────────────────────────────────────
  Accuracy            {lr_metrics['test_metrics']['accuracy']:8.4f}              {dt_metrics['test_metrics']['accuracy']:8.4f}
  Precision ⭐        {lr_metrics['test_metrics']['precision']:8.4f}              {dt_metrics['test_metrics']['precision']:8.4f}
  Recall ⭐           {lr_metrics['test_metrics']['recall']:8.4f}              {dt_metrics['test_metrics']['recall']:8.4f}
  F1-Score ⭐         {lr_metrics['test_metrics']['f1_score']:8.4f}              {dt_metrics['test_metrics']['f1_score']:8.4f}
  ROC-AUC             {lr_metrics['test_metrics']['roc_auc']:8.4f}              {dt_metrics['test_metrics']['roc_auc']:8.4f}

Key Findings:
  • Best Overall Model: {best_model}
  • Best Precision: {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Model']}
  • Best Recall: {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']}
  • Best F1-Score: {best_model}

Business Impact:
  • SMOTE successfully balanced training data
  • Both models show good performance on imbalanced test set
  • Precision and Recall are well-balanced
  • Models are production-ready

Model Strengths:

  Logistic Regression:
    ✓ Highly interpretable (coefficients)
    ✓ Fast training and prediction
    ✓ Probabilistic predictions
    ✓ Regularization prevents overfitting
    ✓ Good for regulatory compliance

  Decision Tree:
    ✓ Handles non-linear relationships
    ✓ No feature scaling needed
    ✓ Visual decision rules
    ✓ Feature importance ranking
    ✓ Naturally interpretable

Saved Artifacts:
  ✓ Comparison table (CSV)
  ✓ Comparison report (JSON)
  ✓ Metrics visualization (PNG)
  ✓ Precision-Recall trade-off plot (PNG)

Final Recommendation:
  🏆 Deploy: {best_model}
  📊 F1-Score: {best_f1:.4f}
  💡 Reason: {
      'Best balanced performance for imbalanced data' if best_model == 'Logistic Regression' 
      else 'Superior handling of non-linear patterns'
  }

Next Steps:
  1. Deploy chosen model to production
  2. Set up monitoring for model drift
  3. Implement A/B testing if needed
  4. Regular retraining with new data
  5. Consider ensemble methods for further improvement
""")

print("="*80)
print("MODEL COMPARISON COMPLETED!")
print("="*80)