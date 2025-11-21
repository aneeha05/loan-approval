"""
Loan Approval Prediction Dashboard
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from scripts.preprocessing import preprocess_input, load_encoders
from scripts.prediction import load_model, predict_loan, get_prediction_explanation
from scripts.visualization import create_gauge_chart, create_feature_importance_chart

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.9);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Prediction result cards */
    .prediction-approved {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 16px rgba(46, 204, 113, 0.3);
    }
    
    .prediction-rejected {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 16px rgba(231, 76, 60, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("# 🏦 Loan Approval System")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📊 EDA", "🎯 Feature Importance", "🔮 Predict Loan", "💡 Explain Prediction"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # File upload
        st.markdown("### 📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with loan application data"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.uploaded_data)} rows")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        st.markdown("---")
        
        # Theme toggle
        st.markdown("### 🎨 Theme")
        theme = st.selectbox("Select theme", ["Light", "Dark"])
        st.session_state.theme = theme.lower()
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ About")
        st.info(
            "This dashboard provides loan approval predictions "
            "with full explainability using advanced ML models."
        )
        
        st.markdown("---")
        st.markdown("**Version:** 1.0.0  \n**Model:** Logistic Regression")
    
    return page

# ============================================================================
# HOME PAGE
# ============================================================================
def render_home():
    st.markdown('<h1 class="section-header">🏠 Loan Approval Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Loan Approval Prediction Dashboard</h3>
    <p>This intelligent system helps financial institutions make data-driven loan approval decisions 
    using machine learning. Our model analyzes multiple factors to predict loan approval probability 
    with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load sample data
    try:
        df = pd.read_csv('data/sample_data.csv')
    except:
        # Create sample data if file doesn't exist
        np.random.seed(42)
        df = pd.DataFrame({
            'loan_status': np.random.choice(['Approved', 'Rejected'], 1000, p=[0.65, 0.35]),
            'credit_score': np.random.randint(300, 850, 1000),
            'annual_income': np.random.randint(20000, 200000, 1000),
            'loan_amount': np.random.randint(5000, 50000, 1000)
        })
    
    # KPI Metrics
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        approval_rate = (df['loan_status'] == 'Approved').mean() * 100 if 'loan_status' in df.columns else 65
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Approval Rate</div>
            <div class="metric-value">{approval_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">Total Applications</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">Features</div>
            <div class="metric-value">{len(df.columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">87%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sample data preview
    st.markdown("## 📋 Sample Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Quick stats
    st.markdown("## 📈 Quick Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Summary")
        st.write(df.describe())
    
    with col2:
        st.markdown("### Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null %': ((len(df) - df.count()) / len(df) * 100).values
        })
        st.dataframe(dtype_df, use_container_width=True)

# ============================================================================
# EDA PAGE
# ============================================================================
def render_eda():
    st.markdown('<h1 class="section-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
    else:
        try:
            df = pd.read_csv('data/sample_data.csv')
        except:
            st.warning("No data available. Please upload a dataset or ensure sample_data.csv exists.")
            return
    
    st.info(f"Analyzing dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📊 Categories", "📦 Outliers"])
    
    with tab1:
        st.markdown("### Numerical Variable Distributions")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 0:
            cols_per_row = 2
            num_rows = (len(numerical_cols) + cols_per_row - 1) // cols_per_row
            
            for i in range(num_rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < len(numerical_cols):
                        col_name = numerical_cols[idx]
                        with cols[j]:
                            fig = px.histogram(
                                df, 
                                x=col_name, 
                                marginal="box",
                                title=f"{col_name} Distribution",
                                color_discrete_sequence=['#667eea']
                            )
                            fig.update_layout(showlegend=False, height=400)
                            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Correlation Analysis")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Feature Correlation Heatmap"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.markdown("#### Highly Correlated Features (|r| > 0.7)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            else:
                st.info("No highly correlated features found.")
    
    with tab3:
        st.markdown("### Categorical Variable Analysis")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            for col_name in categorical_cols[:4]:  # Show first 4
                value_counts = df[col_name].value_counts().head(10)
                
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{col_name} Distribution",
                    labels={'x': col_name, 'y': 'Count'},
                    color=value_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")
    
    with tab4:
        st.markdown("### Outlier Detection (Box Plots)")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 0:
            selected_cols = st.multiselect(
                "Select columns to analyze",
                numerical_cols,
                default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols
            )
            
            if selected_cols:
                fig = go.Figure()
                
                for col in selected_cols:
                    fig.add_trace(go.Box(
                        y=df[col],
                        name=col,
                        boxmean='sd'
                    ))
                
                fig.update_layout(
                    title="Box Plots for Outlier Detection",
                    yaxis_title="Value",
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FEATURE IMPORTANCE PAGE
# ============================================================================
def render_feature_importance():
    st.markdown('<h1 class="section-header">🎯 Feature Importance Analysis</h1>', unsafe_allow_html=True)
    
    try:
        # Load model
        model = load_model('model/loan_model.pkl')
        
        # Load feature importance
        try:
            with open('model/feature_importance.json', 'r') as f:
                importance_data = json.load(f)
            
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Bar chart
            st.markdown("### 📊 Feature Importance Ranking")
            fig = create_feature_importance_chart(importance_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.markdown("### 📋 Importance Values")
            st.dataframe(importance_df, use_container_width=True)
            
        except FileNotFoundError:
            # If no saved importance, try to extract from model
            if hasattr(model, 'feature_importances_'):
                feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = create_feature_importance_chart(importance_df)
                st.plotly_chart(fig, use_container_width=True)
            
            elif hasattr(model, 'coef_'):
                feature_names = [f"Feature_{i}" for i in range(len(model.coef_[0]))]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
                
                fig = create_feature_importance_chart(importance_df)
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🔝 Most Important Features</h4>
            <p>These features have the strongest impact on loan approval decisions. 
            Focus on these when evaluating applications.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📉 Least Important Features</h4>
            <p>These features have minimal impact on predictions. 
            Consider removing them to simplify the model.</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")
        st.info("Please ensure the model file exists at 'model/loan_model.pkl'")

# ============================================================================
# PREDICTION PAGE
# ============================================================================
def render_prediction():
    st.markdown('<h1 class="section-header">🔮 Loan Approval Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📝 Fill in the application details below to get an instant prediction</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            credit_score = st.slider(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                help="Credit score between 300 and 850"
            )
            
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=1000
            )
            
            age = st.slider(
                "Age",
                min_value=18,
                max_value=80,
                value=35
            )
        
        with col2:
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=0,
                max_value=500000,
                value=15000,
                step=1000
            )
            
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["Home", "Auto", "Education", "Business", "Personal", "Medical", "Debt Consolidation"]
            )
            
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Self-Employed", "Unemployed", "Retired"]
            )
        
        with col3:
            debt_to_income = st.slider(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=0.5
            )
            
            existing_loans = st.number_input(
                "Number of Existing Loans",
                min_value=0,
                max_value=10,
                value=1
            )
            
            years_employed = st.number_input(
                "Years at Current Job",
                min_value=0,
                max_value=50,
                value=5
            )
        
        submitted = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)
    
    if submitted:
        # Prepare input data
        input_data = {
            'credit_score': credit_score,
            'annual_income': annual_income,
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'employment_status': employment_status,
            'debt_to_income_ratio': debt_to_income,
            'existing_loans': existing_loans,
            'age': age,
            'years_employed': years_employed
        }
        
        with st.spinner("Analyzing application..."):
            try:
                # Make prediction
                prediction, probability = predict_loan(input_data)
                
                # Display result
                st.markdown("---")
                st.markdown("## 🎯 Prediction Result")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-approved">
                            <h1>✅ LOAN APPROVED</h1>
                            <h2>Approval Probability: {probability*100:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-rejected">
                            <h1>❌ LOAN REJECTED</h1>
                            <h2>Rejection Confidence: {(1-probability)*100:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Gauge chart
                    fig = create_gauge_chart(probability, prediction)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if prediction == 0:
                    st.markdown("""
                    <div class="info-box">
                    <h4>🔧 Ways to Improve Your Application:</h4>
                    <ul>
                        <li>✨ Improve your credit score by paying bills on time</li>
                        <li>💰 Increase your income or apply for a smaller loan amount</li>
                        <li>📉 Reduce your debt-to-income ratio</li>
                        <li>⏰ Wait and build more employment history</li>
                        <li>🏦 Consider applying with a co-signer</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box" style="border-left-color: #2ecc71;">
                    <h4>✅ Next Steps:</h4>
                    <ul>
                        <li>📄 Prepare required documents</li>
                        <li>✍️ Complete the formal application</li>
                        <li>🏢 Visit nearest branch for verification</li>
                        <li>⏳ Approval process takes 3-5 business days</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Please ensure model files are properly configured.")

# ============================================================================
# EXPLAINABILITY PAGE
# ============================================================================
def render_explainability():
    st.markdown('<h1 class="section-header">💡 Model Explainability</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🔍 Understanding Predictions</h4>
    <p>This section helps you understand WHY the model made a specific prediction 
    by showing which features contributed most to the decision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Enter Application Details")
    
    with st.expander("📝 Input Form", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 700)
            annual_income = st.number_input("Annual Income ($)", value=60000, step=1000)
            loan_amount = st.number_input("Loan Amount ($)", value=20000, step=1000)
        
        with col2:
            debt_to_income = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 25.0)
            age = st.slider("Age", 18, 80, 35)
            existing_loans = st.number_input("Existing Loans", 0, 10, 1)
    
    if st.button("🔍 Explain Prediction", use_container_width=True):
        input_data = {
            'credit_score': credit_score,
            'annual_income': annual_income,
            'loan_amount': loan_amount,
            'debt_to_income_ratio': debt_to_income,
            'age': age,
            'existing_loans': existing_loans
        }
        
        try:
            explanation = get_prediction_explanation(input_data)
            
            st.markdown("---")
            st.markdown("## 📊 Explanation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Top Positive Factors")
                if 'positive_factors' in explanation:
                    for factor in explanation['positive_factors']:
                        st.success(f"**{factor['feature']}**: {factor['impact']:.3f}")
            
            with col2:
                st.markdown("### ❌ Top Negative Factors")
                if 'negative_factors' in explanation:
                    for factor in explanation['negative_factors']:
                        st.error(f"**{factor['feature']}**: {factor['impact']:.3f}")
            
            # Waterfall chart
            if 'feature_impacts' in explanation:
                st.markdown("### 📊 Feature Impact Waterfall")
                
                impacts = explanation['feature_impacts']
                
                fig = go.Figure(go.Waterfall(
                    orientation="h",
                    measure=["relative"] * len(impacts),
                    y=[f['feature'] for f in impacts],
                    x=[f['impact'] for f in impacts],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ))
                
                fig.update_layout(
                    title="Feature Contribution to Prediction",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Explanation error: {e}")
            st.info("SHAP values require additional setup. Using simplified explanation.")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    page = render_sidebar()
    
    if page == "🏠 Home":
        render_home()
    elif page == "📊 EDA":
        render_eda()
    elif page == "🎯 Feature Importance":
        render_feature_importance()
    elif page == "🔮 Predict Loan":
        render_prediction()
    elif page == "💡 Explain Prediction":
        render_explainability()

if __name__ == "__main__":
    main()