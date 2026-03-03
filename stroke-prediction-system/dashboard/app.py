"""
Streamlit dashboard for stroke prediction.
"""
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .risk-high {
        color: #ff4b4b;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc00;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def predict(patient_data, explain=False):
    """Make prediction using API."""
    endpoint = f"{API_URL}/predict/explain" if explain else f"{API_URL}/predict"
    
    try:
        response = requests.post(endpoint, json=patient_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.json()['detail']}")
            return None
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return None


def display_risk_gauge(probability):
    """Display risk probability as a gauge."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stroke Risk Probability", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFD700'},
                {'range': [60, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def display_feature_contributions(contributions):
    """Display feature contributions as a bar chart."""
    if not contributions:
        return None
    
    # Sort by absolute value
    sorted_features = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    colors = ['red' if v > 0 else 'green' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='black', width=1)
            )
        )
    ])
    
    fig.update_layout(
        title="Top Feature Contributions to Prediction",
        xaxis_title="Contribution",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )
    
    return fig


# Main app
def main():
    st.title("🏥 Stroke Risk Prediction System")
    st.markdown("### AI-Powered Clinical Decision Support Tool")
    
    # Check API status
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the API server first.")
        st.code("python api/main.py", language="bash")
        return
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        model_info = get_model_info()
        
        if model_info:
            st.metric("Model Type", model_info['model_type'])
            
            if model_info['metrics']:
                st.subheader("Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recall", f"{model_info['metrics']['recall']:.2%}")
                    st.metric("Precision", f"{model_info['metrics']['precision']:.2%}")
                with col2:
                    st.metric("F1-Score", f"{model_info['metrics']['f1_score']:.2%}")
                    st.metric("ROC-AUC", f"{model_info['metrics']['roc_auc']:.2%}")
        
        st.markdown("---")
        st.info("This system uses machine learning to predict stroke risk based on patient data.")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Model Insights", "ℹ️ About"])
    
    with tab1:
        st.header("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=45, help="Patient's age in years")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col2:
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        
        with col3:
            glucose = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
            smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            predict_button = st.button("🔍 Predict Stroke Risk", type="primary", use_container_width=True)
        
        if predict_button:
            # Prepare data
            patient_data = {
                "age": age,
                "gender": gender,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence,
                "avg_glucose_level": glucose,
                "bmi": bmi,
                "smoking_status": smoking
            }
            
            with st.spinner("Analyzing patient data..."):
                result = predict(patient_data, explain=True)
            
            if result:
                st.success("✅ Prediction Complete!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_level = result['risk_level']
                    if risk_level == "High":
                        st.markdown(f'<p class="risk-high">⚠️ {risk_level} Risk</p>', unsafe_allow_html=True)
                    elif risk_level == "Medium":
                        st.markdown(f'<p class="risk-medium">⚡ {risk_level} Risk</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="risk-low">✓ {risk_level} Risk</p>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Stroke Probability", f"{result['probability']:.1%}")
                
                with col3:
                    st.metric("Model Confidence", f"{result['confidence']:.1%}")
                
                # Risk gauge
                st.plotly_chart(display_risk_gauge(result['probability']), use_container_width=True)
                
                # Feature contributions
                if 'feature_contributions' in result:
                    st.subheader("🔍 What's Contributing to This Prediction?")
                    fig = display_feature_contributions(result['feature_contributions'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Top risk factors
                if 'top_risk_factors' in result:
                    st.subheader("📌 Top Risk Factors")
                    for i, factor in enumerate(result['top_risk_factors'][:5], 1):
                        st.write(f"{i}. **{factor['feature']}**: {factor['contribution']:.4f}")
                
                # Clinical recommendation
                st.markdown("---")
                st.subheader("💡 Clinical Recommendation")
                
                if risk_level == "High":
                    st.error("""
                    **High Risk Detected:**
                    - Immediate medical consultation recommended
                    - Consider comprehensive cardiovascular evaluation
                    - Implement lifestyle modifications and risk factor management
                    """)
                elif risk_level == "Medium":
                    st.warning("""
                    **Moderate Risk Detected:**
                    - Regular monitoring advised
                    - Address modifiable risk factors
                    - Schedule follow-up appointment
                    """)
                else:
                    st.success("""
                    **Low Risk:**
                    - Continue healthy lifestyle habits
                    - Regular check-ups as per standard guidelines
                    - Monitor risk factors periodically
                    """)
    
    with tab2:
        st.header("Model Performance Insights")
        
        if model_info and model_info.get('feature_importance'):
            st.subheader("Feature Importance")
            
            importance = model_info['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
            
            df_importance = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
            
            fig = px.bar(
                df_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.info("""
        **About the Model:**
        - Trained on clinical stroke data
        - Optimized for high recall (catching stroke risks)
        - Uses ensemble methods for robust predictions
        - Validated on holdout test set
        """)
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Purpose
        This stroke prediction system uses machine learning to assess stroke risk based on patient health data.
        
        ### 📊 How It Works
        1. **Data Input**: Patient information is collected
        2. **Preprocessing**: Data is cleaned and transformed
        3. **Prediction**: ML model analyzes risk factors
        4. **Explanation**: Feature-contribution estimates show influential factors
        
        ### ⚠️ Important Disclaimer
        This system is for **educational and research purposes only**. It should not replace professional medical advice,
        diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
        
        ### 🔒 Privacy & Security
        - No patient data is stored permanently
        - All predictions are processed in real-time
        - Session data is not logged
        
        ### 📈 Model Performance
        The model has been validated on clinical data with:
        - High recall for detecting stroke risks
        - Balanced performance across different patient demographics
        - Regular updates and monitoring
        
        ### 💻 Technical Stack
        - **Backend**: FastAPI
        - **ML Framework**: Scikit-learn, Imbalanced-learn
        - **Frontend**: Streamlit
        - **Deployment**: Docker
        """)


if __name__ == "__main__":
    main()
