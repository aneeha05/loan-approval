"""
Loan Approval Prediction - Step 1: Exploratory Data Analysis
Purpose: Analyze loan approval dataset and identify patterns
Dataset: loan_approval_dataset.csv from Kaggle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================================
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'visualizations', 'models', 'results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created/verified")

# Create directories first
create_directories()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("LOAN APPROVAL PREDICTION - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load dataset
df = pd.read_csv('loan_approval_dataset.csv')

print("\n1. Dataset Overview")
print("-" * 80)
print(f"Total Records: {len(df):,}")
print(f"Total Features: {len(df.columns)}")
print(f"Dataset Shape: {df.shape}")

# ============================================================================
# 2. BASIC DATA STRUCTURE
# ============================================================================
print("\n2. First 10 Records")
print("-" * 80)
print(df.head(10))

print("\n3. Dataset Information")
print("-" * 80)
df.info()

print("\n4. Column Names and Data Types")
print("-" * 80)
for col in df.columns:
    print(f"{col:40s} - {df[col].dtype}")

print("\n5. Statistical Summary")
print("-" * 80)
print(df.describe())

# ============================================================================
# 3. MISSING VALUES ANALYSIS
# ============================================================================
print("\n6. Missing Values Analysis")
print("-" * 80)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
    'Missing_Percentage', ascending=False
)

if len(missing_data) > 0:
    print("Columns with Missing Values:")
    print(missing_data.to_string(index=False))
    
    # Visualize missing values
    plt.figure(figsize=(10, 6))
    plt.bar(missing_data['Column'], missing_data['Missing_Percentage'], color='coral')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Missing Percentage (%)', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/01_missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Missing values plot saved")
else:
    print("✓ No missing values found!")

# ============================================================================
# 4. IDENTIFY TARGET AND FEATURES
# ============================================================================
print("\n7. Identifying Target Variable")
print("-" * 80)

# Common target column names for loan approval datasets
target_candidates = ['loan_status', 'Loan_Status', 'status', 'approval_status', 'approved']
target_col = None

for candidate in target_candidates:
    if candidate in df.columns:
        target_col = candidate
        break

# If not found, look for columns with binary values that suggest approval/rejection
if not target_col:
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            if len(unique_vals) == 2:
                lower_vals = [str(v).lower() for v in unique_vals]
                if any(term in ' '.join(lower_vals) for term in ['approve', 'reject', 'yes', 'no', 'accept', 'deny']):
                    target_col = col
                    break

if target_col:
    print(f"Target Column: {target_col}")
else:
    print("⚠️ Target column not automatically identified. Please specify manually.")
    # Use the last column as default
    target_col = df.columns[-1]
    print(f"Using default: {target_col}")

# ============================================================================
# 5. TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n8. Target Variable Distribution")
print("-" * 80)

print(f"\nValue Counts for '{target_col}':")
print(df[target_col].value_counts())

print(f"\nPercentage Distribution:")
print(df[target_col].value_counts(normalize=True) * 100)

# Calculate class imbalance
value_counts = df[target_col].value_counts()
if len(value_counts) >= 2:
    majority_class = value_counts.max()
    minority_class = value_counts.min()
    imbalance_ratio = majority_class / minority_class
    
    print(f"\nClass Imbalance Analysis:")
    print(f"  • Majority Class: {majority_class} samples")
    print(f"  • Minority Class: {minority_class} samples")
    print(f"  • Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 1.5:
        print(f"  • ⚠️  Dataset is IMBALANCED - SMOTE will be needed!")
    else:
        print(f"  • ✓ Dataset is relatively balanced")
else:
    imbalance_ratio = 1.0

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], alpha=0.8)
axes[0].set_title(f'{target_col} Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xlabel(target_col, fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Pie chart
colors = ['#2ecc71', '#e74c3c']
df[target_col].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                     colors=colors, startangle=90)
axes[1].set_title(f'{target_col} Distribution (Percentage)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('visualizations/02_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Target distribution plot saved")

# ============================================================================
# 6. NUMERICAL FEATURES ANALYSIS
# ============================================================================
print("\n9. Numerical Features Analysis")
print("-" * 80)

# Separate features from target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numerical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical Features ({len(numerical_cols)}):")
for col in numerical_cols:
    print(f"  • {col}")

if len(numerical_cols) > 0:
    print("\nNumerical Features Statistics:")
    print(X[numerical_cols].describe())
    
    # Distribution plots
    n_plots = min(len(numerical_cols), 9)
    n_rows = (n_plots + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for idx, col in enumerate(numerical_cols[:9]):
        axes[idx].hist(X[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/03_numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Numerical distributions saved")

# ============================================================================
# 7. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n10. Categorical Features Analysis")
print("-" * 80)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical Features ({len(categorical_cols)}):")

if len(categorical_cols) > 0:
    for col in categorical_cols:
        unique_count = X[col].nunique()
        print(f"  • {col:35s}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"    Values: {X[col].value_counts().to_dict()}")
    
    # Visualize categorical distributions
    n_cat_plots = min(len(categorical_cols), 6)
    if n_cat_plots > 0:
        n_rows = (n_cat_plots + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 4))
        axes = axes.flatten() if n_cat_plots > 1 else [axes]
        
        for idx, col in enumerate(categorical_cols[:6]):
            value_counts = X[col].value_counts().head(10)
            axes[idx].bar(range(len(value_counts)), value_counts.values, color='coral', alpha=0.7)
            axes[idx].set_xticks(range(len(value_counts)))
            axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[idx].set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_cat_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/04_categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Categorical distributions saved")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================
print("\n11. Correlation Analysis (Numerical Features)")
print("-" * 80)

if len(numerical_cols) > 1:
    # Correlation matrix
    correlation_matrix = X[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Correlation matrix saved")
    
    # Find highly correlated features
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
    high_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                high_corr.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    if high_corr:
        for feat1, feat2, corr in high_corr:
            print(f"  • {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("  No highly correlated features found (good for model)")

# ============================================================================
# 9. TARGET vs FEATURES ANALYSIS
# ============================================================================
print("\n12. Target Variable Relationship with Features")
print("-" * 80)

# For numerical features - compare distributions by target class
if len(numerical_cols) > 0:
    print("Analyzing numerical features by target class...")
    
    # Select top 4 numerical features
    top_num_features = numerical_cols[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_num_features):
        if idx < len(axes):
            for class_val in df[target_col].unique():
                subset = df[df[target_col] == class_val]
                axes[idx].hist(subset[col].dropna(), alpha=0.6, label=str(class_val), bins=20)
            
            axes[idx].set_title(f'{col} by {target_col}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/06_features_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Features by target class plot saved")

# For categorical features - crosstab analysis
if len(categorical_cols) > 0:
    print("\nCategorical Features vs Target (sample):")
    for col in categorical_cols[:3]:  # Show first 3
        print(f"\n{col} vs {target_col}:")
        crosstab = pd.crosstab(X[col], df[target_col], normalize='index') * 100
        print(crosstab.round(2))

# ============================================================================
# 10. OUTLIER DETECTION
# ============================================================================
print("\n13. Outlier Detection (IQR Method)")
print("-" * 80)

if len(numerical_cols) > 0:
    outlier_summary = []
    
    for col in numerical_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(X)) * 100
        
        if outlier_count > 0:
            outlier_summary.append({
                'Feature': col,
                'Outlier_Count': outlier_count,
                'Outlier_Percentage': outlier_pct
            })
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
        print("Outliers detected:")
        print(outlier_df.to_string(index=False))
    else:
        print("No significant outliers detected")

# ============================================================================
# 11. DATA QUALITY SUMMARY
# ============================================================================
print("\n14. Data Quality Summary")
print("-" * 80)

# Duplicate check
duplicates = df.duplicated().sum()
print(f"Duplicate Rows: {duplicates}")

# Data types
print(f"\nData Types Distribution:")
print(df.dtypes.value_counts())

# Memory usage
print(f"\nMemory Usage:")
print(f"  • Total: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 12. SAVE SUMMARY
# ============================================================================
print("\n15. Saving Analysis Summary")
print("-" * 80)

# Create summary dictionary
summary = {
    'total_records': len(df),
    'total_features': len(df.columns),
    'target_column': target_col,
    'numerical_features': numerical_cols,
    'categorical_features': categorical_cols,
    'missing_values': df.isnull().sum().sum(),
    'duplicates': int(duplicates),
    'class_imbalance_ratio': float(imbalance_ratio),
    'target_distribution': df[target_col].value_counts().to_dict()
}

import json
with open('data/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=4, default=str)
print("✓ Saved: data/eda_summary.json")

# ============================================================================
# 13. FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS SUMMARY")
print("="*80)

print(f"""
Dataset Overview:
  • Total Records: {len(df):,}
  • Total Features: {len(df.columns)}
  • Numerical Features: {len(numerical_cols)}
  • Categorical Features: {len(categorical_cols)}
  
Target Variable: {target_col}
  • Classes: {len(df[target_col].unique())}
  • Distribution: {df[target_col].value_counts().to_dict()}
  • Imbalance Ratio: {imbalance_ratio:.2f}:1
  • Status: {'⚠️  IMBALANCED - SMOTE Required' if imbalance_ratio > 1.5 else '✓ Balanced'}

Data Quality:
  • Missing Values: {df.isnull().sum().sum()}
  • Duplicate Rows: {duplicates}
  • Outliers: {'Detected in some features' if outlier_summary else 'None significant'}

Key Findings:
  ✓ Dataset structure analyzed
  ✓ Target variable identified
  ✓ Class imbalance assessed
  ✓ Feature types categorized
  ✓ Missing values identified
  ✓ Correlations examined

Visualizations Created:
  ✓ 01_missing_values.png
  ✓ 02_target_distribution.png
  ✓ 03_numerical_distributions.png
  ✓ 04_categorical_distributions.png
  ✓ 05_correlation_matrix.png
  ✓ 06_features_by_target.png

Next Steps:
  1. Handle missing values (2_data_preprocessing.py)
  2. Encode categorical features
  3. Split data and apply SMOTE
  4. Train Logistic Regression and Decision Tree models
  5. Evaluate with precision, recall, F1-score
""")

print("="*80)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*80)
