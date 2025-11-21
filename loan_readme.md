# 🏦 Loan Approval Prediction System

A comprehensive machine learning solution for predicting loan approval decisions using **Logistic Regression** and **Decision Tree** classifiers. This project handles imbalanced data using SMOTE and focuses on **Precision, Recall, and F1-Score** metrics critical for credit risk assessment.

## 🎯 Project Overview

Financial institutions need reliable models to:
- **Minimize Risk**: Reduce bad loan approvals (high precision)
- **Maximize Revenue**: Identify good loan opportunities (high recall)
- **Balance Both**: Optimize overall performance (high F1-score)
- **Handle Imbalance**: Deal with skewed class distributions

This system provides end-to-end solution from data exploration to model deployment.

## 📊 Dataset

**Source**: [Loan Approval Dataset from Kaggle](https://www.kaggle.com)

**Features** (typical):
- **Personal**: Number of dependents, education, employment status
- **Financial**: Income, loan amount, loan term, CIBIL score
- **Assets**: Residential, commercial, luxury, bank assets
- **Target**: Loan approval status (Approved/Rejected)

## ✨ Key Features

✅ **Handles Missing Values**: Automatic imputation strategies  
✅ **Categorical Encoding**: Label & One-Hot encoding  
✅ **Class Imbalance**: SMOTE implementation  
✅ **Two Models**: Logistic Regression vs Decision Tree  
✅ **Comprehensive Metrics**: Precision, Recall, F1-Score, ROC-AUC  
✅ **Rich Visualizations**: 10+ plots and analysis charts  
✅ **Model Comparison**: Side-by-side performance evaluation  

## 📁 Project Structure

```
loan-approval-prediction/
│
├── data/                              # Data directory
│   ├── eda_summary.json              # EDA statistics
│   ├── X_encoded.csv                 # Encoded features (unscaled)
│   ├── X_scaled.csv                  # Scaled features
│   ├── y_encoded.csv                 # Encoded target
│   ├── X_train_scaled.csv            # Training (for Logistic Regression)
│   ├── X_train_unscaled.csv          # Training (for Decision Tree)
│   ├── X_test_scaled.csv             # Test set (scaled)
│   ├── X_test_unscaled.csv           # Test set (unscaled)
│   ├── y_train.csv                   # Training target (SMOTE balanced)
│   ├── y_test.csv                    # Test target (original)
│   └── split_info.json               # Split metadata
│
├── models/                            # Saved models
│   ├── logistic_regression_model.pkl # Trained LR model
│   ├── decision_tree_model.pkl       # Trained DT model
│   ├── feature_encoders.pkl          # Categorical encoders
│   ├── target_encoder.pkl            # Target encoder
│   ├── feature_scaler.pkl            # StandardScaler
│   ├── numerical_imputer.pkl         # Imputer for numeric features
│   ├── categorical_imputer.pkl       # Imputer for categorical features
│   └── feature_info.json             # Feature metadata
│
├── results/                           # Model outputs
│   ├── logistic_regression_performance.json
│   ├── decision_tree_performance.json
│   ├── model_comparison_table.csv
│   ├── model_comparison_report.json
│   ├── logistic_regression_predictions.csv
│   ├── decision_tree_predictions.csv
│   ├── logistic_regression_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── logistic_regression_roc_pr_curves.png
│   ├── decision_tree_roc_pr_curves.png
│   ├── logistic_regression_feature_importance.png
│   ├── decision_tree_feature_importance.png
│   ├── decision_tree_structure.png
│   ├── model_comparison_metrics.png
│   └── precision_recall_tradeoff.png
│
├── visualizations/                    # EDA plots
│   ├── 01_missing_values.png
│   ├── 02_target_distribution.png
│   ├── 03_numerical_distributions.png
│   ├── 04_categorical_distributions.png
│   ├── 05_correlation_matrix.png
│   ├── 06_features_by_target.png
│   └── 07_smote_comparison.png
│
├── 1_exploratory_analysis.py          # Step 1: EDA
├── 2_data_preprocessing.py            # Step 2: Preprocessing
├── 3_train_test_split_smote.py        # Step 3: Split & SMOTE
├── 4_logistic_regression.py           # Step 4: Train LR
├── 5_decision_tree.py                 # Step 5: Train DT
├── 6_model_comparison.py              # Step 6: Compare models
├── requirements.txt                   # Dependencies
├── README.md                          # This file
└── loan_approval_dataset.csv          # Original dataset
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
```bash
cd loan-approval-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create required directories**
```bash
mkdir data models results visualizations
```

4. **Place your dataset**
- Download `loan_approval_dataset.csv` from Kaggle
- Place it in the project root directory

### Running the Pipeline

Execute scripts **in order**:

```bash
# Step 1: Exploratory Data Analysis
python 1_exploratory_analysis.py

# Step 2: Data Preprocessing
python 2_data_preprocessing.py

# Step 3: Train-Test Split & SMOTE
python 3_train_test_split_smote.py

# Step 4: Train Logistic Regression
python 4_logistic_regression.py

# Step 5: Train Decision Tree
python 5_decision_tree.py

# Step 6: Compare Models
python 6_model_comparison.py
```

## 📈 Model Performance

### Typical Results

| Metric | Logistic Regression | Decision Tree |
|--------|---------------------|---------------|
| **Accuracy** | 0.85-0.88 | 0.83-0.87 |
| **Precision** ⭐ | 0.82-0.86 | 0.80-0.85 |
| **Recall** ⭐ | 0.80-0.84 | 0.82-0.86 |
| **F1-Score** ⭐ | 0.81-0.85 | 0.81-0.85 |
| **ROC-AUC** | 0.88-0.92 | 0.85-0.90 |

*Note: Results vary based on dataset*

### Key Metrics Explained

#### Precision (Positive Predictive Value)
- **Definition**: Of all predicted approvals, what % are actually good loans?
- **Formula**: TP / (TP + FP)
- **Business Impact**: HIGH precision = Fewer bad loans approved = Lower risk
- **Example**: 85% precision means 15% of approved loans may default

#### Recall (Sensitivity, True Positive Rate)
- **Definition**: Of all actual good loans, what % did we approve?
- **Formula**: TP / (TP + FN)
- **Business Impact**: HIGH recall = Catch more opportunities = More revenue
- **Example**: 80% recall means we miss 20% of good loan opportunities

#### F1-Score (Harmonic Mean)
- **Definition**: Balanced measure of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Business Impact**: Best single metric for imbalanced data
- **Sweet Spot**: Balances risk and opportunity

## 🎯 Handling Imbalanced Data

### Problem
Real-world loan datasets are often imbalanced:
- **Approved loans**: 60-70% (majority class)
- **Rejected loans**: 30-40% (minority class)

### Solution: SMOTE (Synthetic Minority Over-sampling Technique)

**How it works**:
1. Identifies minority class samples
2. Creates synthetic samples by interpolating between neighbors
3. Balances the training set to 50-50

**Why SMOTE?**
- ✓ Better than random oversampling (no exact duplicates)
- ✓ Better than undersampling (no data loss)
- ✓ Creates realistic synthetic examples
- ✓ Improves minority class detection

**Important**: SMOTE is applied **only to training data**, never to test data!

### Before vs After SMOTE

```
Before SMOTE:
  Class 0 (Rejected): 400 samples (40%)
  Class 1 (Approved): 600 samples (60%)
  Imbalance Ratio: 1.5:1

After SMOTE (Training Only):
  Class 0 (Rejected): 600 samples (50%)
  Class 1 (Approved): 600 samples (50%)
  Imbalance Ratio: 1:1

Test Set (Unchanged):
  Class 0 (Rejected): 100 samples (40%)
  Class 1 (Approved): 150 samples (60%)
  Imbalance Ratio: 1.5:1
```

## 🤖 Model Comparison

### Logistic Regression

**Strengths**:
- ✓ Highly interpretable (coefficients show impact)
- ✓ Fast training and prediction
- ✓ Probabilistic output (confidence scores)
- ✓ Works well with scaled features
- ✓ Less prone to overfitting
- ✓ Good for regulatory compliance

**Best For**:
- Linear relationships
- Explainability requirements
- Fast deployment
- When feature scaling is acceptable

### Decision Tree

**Strengths**:
- ✓ Handles non-linear relationships naturally
- ✓ No feature scaling required
- ✓ Visual decision rules (tree diagram)
- ✓ Feature importance via Gini
- ✓ Captures complex interactions
- ✓ Intuitive business logic

**Best For**:
- Non-linear patterns
- Mixed data types
- Quick prototyping
- Visual explanations

### When to Choose Which?

| Scenario | Recommended Model |
|----------|-------------------|
| Need interpretability for regulators | **Logistic Regression** |
| Complex non-linear relationships | **Decision Tree** |
| Fast deployment required | **Logistic Regression** |
| No time for feature scaling | **Decision Tree** |
| Highest precision needed | **Check comparison results** |
| Highest recall needed | **Check comparison results** |
| Best F1-score | **Check comparison results** |

## 📊 Visualizations

The system generates **15+ visualizations**:

### EDA Phase
1. Missing values analysis
2. Target distribution (imbalance check)
3. Numerical feature distributions
4. Categorical feature distributions
5. Correlation matrix
6. Features by target class

### Model Training Phase
7. SMOTE comparison (before/after)
8. Confusion matrices (both models)
9. ROC curves
10. Precision-Recall curves
11. Feature importance (both models)
12. Decision tree structure

### Comparison Phase
13. Metric comparison bars
14. Precision-Recall trade-off
15. Model characteristics table

## 💼 Business Applications

### Risk Management
- **High Precision Strategy**: Minimize bad loans
  - Set threshold to favor precision
  - Accept lower recall for safety
  - Suitable for conservative lenders

### Growth Strategy
- **High Recall Strategy**: Maximize approvals
  - Set threshold to favor recall
  - Accept more risk for growth
  - Suitable for competitive markets

### Balanced Strategy
- **High F1-Score**: Optimize both metrics
  - Use default 0.5 threshold
  - Balance risk and opportunity
  - Most common approach

## 🔍 Model Interpretability

### Logistic Regression Coefficients

```python
Top Features (Example):
1. CIBIL_Score:        +0.85 (Strong positive impact)
2. Income_Annum:       +0.62 (Moderate positive)
3. Loan_Amount:        -0.45 (Moderate negative)
4. Self_Employed_No:   +0.38 (Positive if not self-employed)
```

**Interpretation**: 
- Each unit increase in CIBIL score increases log-odds of approval by 0.85
- Higher income increases approval chances
- Larger loan amounts decrease approval probability

### Decision Tree Rules

```
Example Decision Path:
IF CIBIL_Score >= 700
  AND Income_Annum >= 500000
    AND Loan_Amount <= 2000000
      THEN Approve (confidence: 92%)
```

## 🛠️ Advanced Usage

### Threshold Tuning

Adjust the decision threshold for different business needs:

```python
import pickle
import numpy as np

# Load model
with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Conservative (high precision): threshold = 0.6
y_pred_conservative = (y_proba >= 0.6).astype(int)

# Aggressive (high recall): threshold = 0.4
y_pred_aggressive = (y_proba >= 0.4).astype(int)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X_train, y_train, 
                        cv=5, scoring='f1')
print(f"CV F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Feature Engineering Ideas

Add derived features to improve performance:

```python
# Debt-to-Income Ratio
df['debt_to_income'] = df['loan_amount'] / df['income_annum']

# Asset-to-Loan Ratio
df['asset_to_loan'] = df['total_assets'] / df['loan_amount']

# CIBIL Categories
df['cibil_category'] = pd.cut(df['cibil_score'], 
                               bins=[0, 600, 700, 800, 900],
                               labels=['Poor', 'Fair', 'Good', 'Excellent'])
```

## 📝 Best Practices

1. **Always apply SMOTE only to training data**
2. **Use stratified split to maintain class distribution**
3. **Focus on precision, recall, F1 - not just accuracy**
4. **Check for overfitting (train vs test performance)**
5. **Monitor false positives (business risk)**
6. **Monitor false negatives (lost opportunities)**
7. **Regularly retrain with new data**
8. **Set up model monitoring in production**

## 🐛 Troubleshooting

### Common Issues

**Issue**: Low recall on test set
- **Solution**: Check if SMOTE was applied correctly to training only
- **Solution**: Try adjusting decision threshold lower

**Issue**: High training accuracy, low test accuracy
- **Solution**: Model is overfitting - reduce Decision Tree depth
- **Solution**: Add regularization to Logistic Regression

**Issue**: Both precision and recall are low
- **Solution**: Features may not be informative enough
- **Solution**: Try feature engineering or collect more data

**Issue**: Imbalanced-learn import error
- **Solution**: `pip install imbalanced-learn`

## 🤝 Contributing

Improvements welcome:
- Additional models (Random Forest, XGBoost, etc.)
- Ensemble methods
- Hyperparameter tuning (GridSearchCV)
- Cost-sensitive learning
- Web interface for predictions

## 📄 License

This project is for educational and commercial use.

## 👥 Author

Credit Risk Modeling Team

## 🙏 Acknowledgments

- Kaggle for the loan approval dataset
- scikit-learn for ML algorithms
- imbalanced-learn for SMOTE implementation
- Community for best practices

---

**Ready to deploy intelligent loan approval decisions! 🚀**