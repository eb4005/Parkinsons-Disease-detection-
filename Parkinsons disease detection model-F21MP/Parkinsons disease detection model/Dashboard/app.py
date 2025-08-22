import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
            


    .positive-prediction {
            
     background-color: rgba(255, 235, 235, 0.5); /* Lighter red with 50% opacity */
     border-left: 5px solid #f44336;
    }
    .negative-prediction {
        background-color: rgba(232, 245, 232, 0.5); /* Lighter green with 50% opacity */
        border-left: 5px solid #4caf50;
    }
    .feature-importance {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model('parkinsons.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        st.info("Please make sure 'parkinsons.keras' and 'scaler.pkl' files are in the same directory as this script.")
        return None, None

FEATURE_INFO = {
    'age': {
        'name': 'Age',
        'description': 'Age of the patient in years',
        'range': (18, 100),
        'default': 65
    },
    'sex': {
        'name': 'Sex',
        'description': 'Gender of the patient (0: Female, 1: Male)',
        'range': (0, 1),
        'default': 0
    },
    'test_time': {
        'name': 'Test Time',
        'description': 'Time since first test in days',
        'range': (0, 1000),
        'default': 100
    },
    'Jitter(%)': {
        'name': 'Jitter (%)',
        'description': 'Frequency variation in voice',
        'range': (0.0, 1.0),
        'default': 0.006
    },
    'Jitter(Abs)': {
        'name': 'Jitter (Absolute)',
        'description': 'Absolute jitter in voice',
        'range': (0.0, 0.001),
        'default': 0.00004
    },
    'Jitter:RAP': {
        'name': 'Jitter RAP',
        'description': 'Relative average perturbation',
        'range': (0.0, 0.1),
        'default': 0.003
    },
    'Jitter:PPQ5': {
        'name': 'Jitter PPQ5',
        'description': 'Five-point period perturbation quotient',
        'range': (0.0, 0.1),
        'default': 0.003
    },
    'Jitter:DDP': {
        'name': 'Jitter DDP',
        'description': 'Average absolute difference of differences',
        'range': (0.0, 0.1),
        'default': 0.009
    },
    'Shimmer': {
        'name': 'Shimmer',
        'description': 'Amplitude variation in voice',
        'range': (0.0, 1.0),
        'default': 0.03
    },
    'Shimmer(dB)': {
        'name': 'Shimmer (dB)',
        'description': 'Shimmer in decibels',
        'range': (0.0, 2.0),
        'default': 0.3
    },
    'Shimmer:APQ3': {
        'name': 'Shimmer APQ3',
        'description': 'Three-point amplitude perturbation quotient',
        'range': (0.0, 0.1),
        'default': 0.015
    },
    'Shimmer:APQ5': {
        'name': 'Shimmer APQ5',
        'description': 'Five-point amplitude perturbation quotient',
        'range': (0.0, 0.1),
        'default': 0.017
    },
    'Shimmer:APQ11': {
        'name': 'Shimmer APQ11',
        'description': 'Eleven-point amplitude perturbation quotient',
        'range': (0.0, 0.1),
        'default': 0.024
    },
    'Shimmer:DDA': {
        'name': 'Shimmer DDA',
        'description': 'Average absolute difference of differences',
        'range': (0.0, 0.1),
        'default': 0.045
    },
    'NHR': {
        'name': 'NHR',
        'description': 'Noise-to-harmonics ratio',
        'range': (0.0, 1.0),
        'default': 0.02
    },
    'HNR': {
        'name': 'HNR',
        'description': 'Harmonics-to-noise ratio',
        'range': (0.0, 50.0),
        'default': 22.0
    },
    'RPDE': {
        'name': 'RPDE',
        'description': 'Recurrence period density entropy',
        'range': (0.0, 1.0),
        'default': 0.5
    },
    'DFA': {
        'name': 'DFA',
        'description': 'Detrended fluctuation analysis',
        'range': (0.0, 1.0),
        'default': 0.7
    },
    'PPE': {
        'name': 'PPE',
        'description': 'Pitch period entropy',
        'range': (0.0, 1.0),
        'default': 0.2
    }
}

def main():
    st.markdown('<h1 class="main-header"> Parkinson\'s Disease Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.stop()
    
    
    st.sidebar.title(" Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction"])
    
    if page == "Prediction":
        prediction_page(model, scaler)
    

def prediction_page(model, scaler):
    st.markdown('<h2 class="sub-header"> Make a Prediction</h2>', unsafe_allow_html=True)
    
   
    input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV File"])
    
    if input_method == "Manual Input":
        manual_input_prediction(model, scaler)
    else:
        csv_upload_prediction(model, scaler)

def manual_input_prediction(model, scaler):
    st.markdown("### Enter Patient Voice Features:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        inputs = {}
        
       
        feature_list = list(FEATURE_INFO.keys())
        for i, feature in enumerate(feature_list):
            info = FEATURE_INFO[feature]
            col = [col1, col2, col3][i % 3]
            
            with col:
                if feature == 'sex':
                    inputs[feature] = st.selectbox(
                        f"{info['name']}",
                        options=[0, 1],
                        format_func=lambda x: "Female" if x == 0 else "Male",
                        index=info['default'],
                        help=info['description']
                    )
                else:
                    inputs[feature] = st.number_input(
                        f"{info['name']}",
                        min_value=float(info['range'][0]),
                        max_value=float(info['range'][1]),
                        value=float(info['default']),
                        help=info['description'],
                        format="%.6f" if info['range'][1] < 1 else "%.2f"
                    )
        
        submitted = st.form_submit_button(" Predict", use_container_width=True)
        
        if submitted:
            # Make prediction
            prediction, probability = make_prediction(inputs, model, scaler)
            display_prediction_results(prediction, probability, inputs)

def csv_upload_prediction(model, scaler):
    st.markdown("### Upload CSV File:")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview:")
            st.dataframe(df.head())
            
            
            required_columns = list(FEATURE_INFO.keys())
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            if st.button(" Predict All", use_container_width=True):
                predictions = []
                probabilities = []
                
                for _, row in df.iterrows():
                    inputs = {col: row[col] for col in required_columns}
                    pred, prob = make_prediction(inputs, model, scaler)
                    predictions.append(pred)
                    probabilities.append(prob)
                
                
                df['Prediction'] = predictions
                df['Probability'] = probabilities
                df['Risk_Level'] = df['Probability'].apply(get_risk_level)
                
                # Display results
                st.write("### Prediction Results:")
                st.dataframe(df[['Prediction', 'Probability', 'Risk_Level']])
                
                # Summary 
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", len(df))
                with col2:
                    st.metric("High Risk", sum(predictions))
                with col3:
                    st.metric("Low Risk", len(predictions) - sum(predictions))
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    label=" Download Results",
                    data=csv,
                    file_name="parkinsons_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def make_prediction(inputs, model, scaler):
    
    feature_array = np.array([[inputs[feature] for feature in FEATURE_INFO.keys()]])
    
    
    scaled_features = scaler.transform(feature_array)
    
    
    scaled_features_cnn = scaled_features.reshape((scaled_features.shape[0], scaled_features.shape[1], 1))
    
   
    probability = model.predict(scaled_features_cnn)[0][0]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def display_prediction_results(prediction, probability, inputs):
    st.markdown("---")
    st.markdown("##  Prediction Results")
    
    
    if prediction == 1:
        st.markdown(f'''
        <div class="prediction-box positive-prediction">
            <h3> HIGH RISK for Parkinson's Disease</h3>
            <p><strong>Probability:</strong> {probability:.1%}</p>
            <p>The model indicates a high likelihood of Parkinson's disease based on the voice features.</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="prediction-box negative-prediction">
            <h3>LOW RISK for Parkinson's Disease</h3>
            <p><strong>Probability:</strong> {probability:.1%}</p>
            <p>The model indicates a low likelihood of Parkinson's disease based on the voice features.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    
    risk_level = get_risk_level(probability)
    st.markdown(f"**Risk Level:** {risk_level}")
    
    # Probability gauge indicator
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature visualization
    display_feature_analysis(inputs, probability)

def get_risk_level(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def display_feature_analysis(inputs, probability):
    st.markdown("###  Feature Analysis")
    
    
    feature_names = list(FEATURE_INFO.keys())
    feature_values = [inputs[feature] for feature in feature_names]
    
    # Radar chartt
    voice_features = ['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
    voice_values = [inputs[feature] for feature in voice_features if feature in inputs]
    
    if len(voice_values) == len(voice_features):
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=voice_values,
            theta=voice_features,
            fill='toself',
            name='Patient Values'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(voice_values) * 1.2]
                )),
            showlegend=True,
            title="Voice Feature Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    

    st.markdown("###  Input Feature Values")
    feature_df = pd.DataFrame([
        {
            'Feature': FEATURE_INFO[feature]['name'],
            'Value': inputs[feature],
            'Description': FEATURE_INFO[feature]['description']
        }
        for feature in feature_names
    ])
    
    st.dataframe(feature_df, use_container_width=True)


if __name__ == "__main__":
    main()


