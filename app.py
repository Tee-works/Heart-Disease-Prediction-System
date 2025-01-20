"""
Enhanced Streamlit app for Heart Disease Prediction with additional features
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    """Load the trained model and scaler."""
    model = joblib.load('models/heart_disease_predictor_model.joblib')
    scaler = joblib.load('models/heart_disease_predictor_scaler.joblib')
    return model, scaler

def plot_risk_factors(input_data, feature_importance):
    """Create visualization of patient's risk factors."""
    # Prepare data for visualization
    normalized_values = pd.Series(input_data[0])
    top_features = feature_importance.head()
    
    # Create risk factor visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if v > 0.5 else 'blue' for v in normalized_values]
    sns.barplot(x=top_features['Importance'], y=top_features['Feature'])
    plt.title('Top Risk Factors')
    return fig

def calculate_risk_breakdown(input_data, feature_importance):
    """Calculate detailed risk breakdown."""
    risk_factors = []
    for feature, importance in zip(feature_importance['Feature'], feature_importance['Importance']):
        if importance > 0.1:  # Only consider significant factors
            risk_factors.append({
                'Factor': feature,
                'Contribution': importance,
                'Status': 'High Risk' if importance > 0.15 else 'Moderate Risk'
            })
    return pd.DataFrame(risk_factors)

def provide_recommendations(prediction, risk_factors):
    """Provide personalized medical recommendations."""
    recommendations = []
    
    if prediction > 0.5:
        recommendations.append("🏥 Immediate consultation with a cardiologist recommended")
        recommendations.append("❤️ Regular blood pressure monitoring")
        recommendations.append("🏃 Supervised exercise program consideration")
    else:
        recommendations.append("✅ Continue regular health monitoring")
        recommendations.append("🥗 Maintain heart-healthy lifestyle")
    
    # Add specific recommendations based on risk factors
    for _, factor in risk_factors.iterrows():
        if factor['Status'] == 'High Risk':
            if 'Cholesterol' in factor['Factor']:
                recommendations.append("🥑 Dietary modification recommended")
            elif 'Blood Pressure' in factor['Factor']:
                recommendations.append("🧂 Reduce sodium intake")
            elif 'Exercise' in factor['Factor']:
                recommendations.append("👟 Gradual increase in physical activity")
    
    return recommendations

def main():
    st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title('Heart Disease Risk Prediction System')
    st.write('Enter patient information for heart disease risk assessment')
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Risk Assessment", "Information"])
    
    with tab1:
        # Input form in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=20, max_value=100, value=50,
                                help="Patient's age in years")
            sex = st.selectbox('Sex', ['Male', 'Female'])
            cp = st.selectbox('Chest Pain Type', 
                            ['Typical Angina', 'Atypical Angina', 
                             'Non-anginal Pain', 'Asymptomatic'],
                            help="Type of chest pain experienced")
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 
                                     min_value=90, max_value=200, value=120,
                                     help="Blood pressure when at rest")
            chol = st.number_input('Serum Cholesterol (mg/dl)', 
                                 min_value=100, max_value=600, value=200,
                                 help="Cholesterol level in blood")
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', 
                             ['No', 'Yes'],
                             help="Blood sugar level after fasting")
            
        with col2:
            restecg = st.selectbox('Resting ECG Results', 
                                 ['Normal', 'ST-T Wave Abnormality', 
                                  'Left Ventricular Hypertrophy'],
                                 help="ECG results when at rest")
            thalach = st.number_input('Maximum Heart Rate', 
                                    min_value=60, max_value=220, value=150,
                                    help="Maximum heart rate achieved")
            exang = st.selectbox('Exercise Induced Angina', 
                               ['No', 'Yes'],
                               help="Chest pain during exercise")
            oldpeak = st.number_input('ST Depression', 
                                    min_value=0.0, max_value=6.0, value=0.0,
                                    help="ST depression induced by exercise")
            slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                               ['Upsloping', 'Flat', 'Downsloping'])
            ca = st.number_input('Number of Major Vessels', 
                               min_value=0, max_value=3, value=0)
            thal = st.selectbox('Thalassemia', 
                              ['Normal', 'Fixed Defect', 'Reversible Defect'])
        
        if st.button('Calculate Risk', use_container_width=True):
            # Convert inputs
            sex = 1 if sex == 'Male' else 0
            cp = {'Typical Angina': 0, 'Atypical Angina': 1, 
                 'Non-anginal Pain': 2, 'Asymptomatic': 3}[cp]
            fbs = 1 if fbs == 'Yes' else 0
            restecg = {'Normal': 0, 'ST-T Wave Abnormality': 1, 
                      'Left Ventricular Hypertrophy': 2}[restecg]
            exang = 1 if exang == 'Yes' else 0
            slope = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}[slope]
            thal = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}[thal]
            
            # Create input array
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, ca, thal]])
            
            # Make prediction
            model, scaler = load_model()
            input_scaled = scaler.transform(input_data)
            prediction = model.predict_proba(input_scaled)[0]
            
            # Display results in three columns
            st.subheader('Assessment Results')
            res_col1, res_col2, res_col3 = st.columns([2,2,3])
            
            with res_col1:
                st.metric('Heart Disease Risk', f'{prediction[1]:.1%}')
                if prediction[1] > 0.5:
                    st.error('⚠️ High Risk: Further evaluation recommended')
                else:
                    st.success('✅ Low Risk: Regular monitoring advised')
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 
                           'Cholesterol', 'Blood Sugar', 'ECG', 'Max Heart Rate',
                           'Exercise Angina', 'ST Depression', 'ST Slope', 
                           'Vessels', 'Thalassemia'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            with res_col2:
                st.subheader('Risk Analysis')
                risk_breakdown = calculate_risk_breakdown(input_data, feature_importance)
                st.dataframe(risk_breakdown, hide_index=True)
            
            with res_col3:
                st.subheader('Recommendations')
                recommendations = provide_recommendations(prediction[1], risk_breakdown)
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            
            # Visualization
            st.subheader('Risk Factor Visualization')
            risk_plot = plot_risk_factors(input_data, feature_importance)
            st.pyplot(risk_plot)
    
    with tab2:
        st.subheader("About Heart Disease Risk Factors")
        st.write("""
        This tool assesses heart disease risk based on multiple clinical factors:
        
        • Age and Sex: Basic demographic factors that influence heart disease risk
        • Chest Pain: Different types indicate varying levels of heart disease risk
        • Blood Pressure: A key indicator of cardiovascular health
        • Cholesterol: High levels can indicate increased heart disease risk
        • ECG Results: Electrical activity of the heart can show abnormalities
        • Exercise Response: How the heart responds to physical stress
        
        The assessment combines these factors using machine learning to estimate overall risk.
        """)
        
        st.warning("""
        Note: This tool is meant to assist medical professionals and should not 
        replace clinical judgment. Always consult with healthcare providers for 
        medical decisions.
        """)

if __name__ == '__main__':
    main()