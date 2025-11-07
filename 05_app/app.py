"""
Interactive Crystallization Component Predictor
===============================================
Streamlit app to test Simple Baseline and Advanced Baseline models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crystallization Predictor",
    page_icon="🔬",
    layout="wide"
)

# Title
st.title("🔬 Crystallization Component Predictor")
st.markdown("### Predict crystallization components using Machine Learning")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Selection")
approach = st.sidebar.radio(
    "Choose Approach:",
    ["Advanced Baseline (Recommended)", "Simple Baseline"],
    help="Advanced has concentration parsing and better accuracy"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Performance")

# Load and display results
try:
    with open('../02_models/simple_baseline/training_results.json', 'r') as f:
        simple_results = json.load(f)
    with open('../02_models/advanced_baseline/training_results.json', 'r') as f:
        advanced_results = json.load(f)
    
    if "Simple" in approach:
        st.sidebar.metric("Name Accuracy", "61.12%")
        st.sidebar.metric("pH R²", "95.58%")
        st.sidebar.warning("Conc: N/A")
    else:
        st.sidebar.metric("Name Accuracy", "64.18%")
        st.sidebar.metric("Conc R²", "47.33%")
        st.sidebar.metric("pH R²", "99.34%")
        st.sidebar.success("All working!")
        
except Exception as e:
    st.sidebar.error(f"Could not load results: {e}")

# Input Form
st.header("🎯 Input Crystallization Parameters")

col1, col2 = st.columns(2)

with col1:
    cryst_method = st.selectbox(
        "Crystallization Method",
        [
            "VAPOR DIFFUSION, SITTING DROP",
            "VAPOR DIFFUSION, HANGING DROP",
            "VAPOR DIFFUSION",
            "BATCH MODE",
            "MICROBATCH"
        ]
    )
    
    temp = st.slider("Temperature (K)", 250.0, 320.0, 293.0, 1.0)
    ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)

with col2:
    matthews = st.slider("Matthews Coefficient", 1.0, 4.5, 2.2, 0.1)
    solvent = st.slider("Percent Solvent Content (%)", 0.0, 100.0, 45.0, 1.0)

st.markdown("---")

# Predict button
if st.button("🚀 Predict Components", type="primary", use_container_width=True):
    
    try:
        with st.spinner("Making predictions..."):
            
            if "Advanced" in approach:
                # Load advanced models
                model_name = joblib.load('../02_models/advanced_baseline/model_component_name.pkl')
                model_conc = joblib.load('../02_models/advanced_baseline/model_component_conc.pkl')
                model_ph = joblib.load('../02_models/advanced_baseline/model_component_ph.pkl')
                le = joblib.load('../02_models/advanced_baseline/label_encoder_name.pkl')
                scaler = joblib.load('../02_models/advanced_baseline/scaler.pkl')
                tfidf = joblib.load('../02_models/advanced_baseline/tfidf.pkl')
                
                # Feature engineering (Advanced Baseline needs 8 features)
                temp_ph_int = temp * ph
                matthews_solvent_int = matthews * solvent
                ph_diff = 0  # Unknown for new prediction
                solvent_ratio = solvent / (matthews + 1e-6)
                
                numerical = np.array([[temp, ph, matthews, solvent,
                                      temp_ph_int, matthews_solvent_int,
                                      ph_diff, solvent_ratio]])
                
            else:
                # Load simple models
                model_name = joblib.load('../02_models/simple_baseline/model_component_name.pkl')
                model_ph = joblib.load('../02_models/simple_baseline/model_component_ph.pkl')
                le = joblib.load('../02_models/simple_baseline/label_encoder_name.pkl')
                scaler = joblib.load('../02_models/simple_baseline/scaler.pkl')
                tfidf = joblib.load('../02_models/simple_baseline/tfidf.pkl')
                
                # Simple baseline: only 4 features
                numerical = np.array([[temp, ph, matthews, solvent]])
            
            # Scale numerical
            numerical_scaled = scaler.transform(numerical)
            
            # TF-IDF for method
            method_tfidf = tfidf.transform([cryst_method.upper()]).toarray()
            
            # Combine
            X_pred = np.concatenate([numerical_scaled, method_tfidf], axis=1)
            
            # Predictions
            pred_name_idx = model_name.predict(X_pred)[0]
            pred_name = le.inverse_transform([pred_name_idx])[0]
            pred_name_proba = model_name.predict_proba(X_pred)[0]
            top_5_idx = np.argsort(pred_name_proba)[-5:][::-1]
            top_5_names = le.inverse_transform(top_5_idx)
            top_5_proba = pred_name_proba[top_5_idx]
            
            pred_ph = model_ph.predict(X_pred)[0]
            
            if "Advanced" in approach:
                pred_conc = model_conc.predict(X_pred)[0]
            
        # Display Results
        st.success("✅ Predictions Complete!")
        st.markdown("---")
        
        st.header("📊 Prediction Results")
        
        # Component Name
        st.subheader("1️⃣ Component_1_Name")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Predicted Component", pred_name)
        
        with col2:
            st.markdown("**Top 5 Predictions:**")
            top5_df = pd.DataFrame({
                'Rank': range(1, 6),
                'Component': top_5_names,
                'Probability': [f"{p:.2%}" for p in top_5_proba]
            })
            st.dataframe(top5_df, hide_index=True, use_container_width=True)
        
        # Concentration
        st.subheader("2️⃣ Component_1_Conc")
        if "Advanced" in approach:
            st.metric("Predicted Concentration", f"{pred_conc:.4f} M (log-scale)")
            st.info(f"Actual Molarity: {10**pred_conc:.6f} M")
        else:
            st.warning("Not available in Simple Baseline")
        
        # pH
        st.subheader("3️⃣ Component_1_pH")
        st.metric("Predicted pH", f"{pred_ph:.2f}")
        
        # pH visualization
        ph_percent = (pred_ph / 14) * 100
        ph_color = "red" if pred_ph < 6 else ("green" if pred_ph < 8 else "blue")
        st.markdown(f"""
        <div style='background: linear-gradient(to right, red, yellow, green, cyan, blue); 
                    height: 30px; border-radius: 5px; margin: 10px 0;'></div>
        <div style='display: flex; justify-content: space-between;'>
            <span>0 (Acidic)</span>
            <span>7 (Neutral)</span>
            <span>14 (Basic)</span>
        </div>
        <div style='text-align: center; margin-top: 10px;'>
            <b style='font-size: 20px; color: {ph_color};'>pH = {pred_ph:.2f}</b>
        </div>
        """, unsafe_allow_html=True)
        
        # Input Summary
        st.markdown("---")
        st.subheader("📥 Input Summary")
        input_df = pd.DataFrame({
            'Feature': ['Crystallization Method', 'Temperature', 'pH', 'Matthews Coefficient', 'Solvent Content'],
            'Value': [cryst_method, f"{temp} K", f"{ph:.1f}", f"{matthews:.2f}", f"{solvent:.1f}%"]
        })
        st.table(input_df)
        
    except FileNotFoundError as e:
        st.error(f"""
        ❌ Model files not found!
        
        Error: {e}
        
        Please ensure models are in:
        - ../02_models/simple_baseline/
        - ../02_models/advanced_baseline/
        """)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())

# Comparison
st.markdown("---")
st.header("📈 Approach Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Simple Baseline", "61.12%", help="Name accuracy")
    st.caption("Fast & Simple")
    st.caption("No concentration")

with col2:
    st.metric("Advanced Baseline", "64.18%", delta="BEST", help="Name accuracy")
    st.caption("Concentration: R²=0.47")
    st.caption("pH: R²=0.99")

with col3:
    st.metric("Transformer", "53.85%", help="Name accuracy")  
    st.caption("Deep learning")
    st.caption("Needs more data")

# Visualizations
st.markdown("---")
st.header("📊 Performance Visualizations")

try:
    tab1, tab2, tab3, tab4 = st.tabs(["Name Accuracy", "Conc R²", "pH R²", "Complete"])
    
    with tab1:
        st.image('../04_results/visualizations/01_component_name_comparison.png')
    with tab2:
        st.image('../04_results/visualizations/02_component_conc_comparison.png')
    with tab3:
        st.image('../04_results/visualizations/03_component_ph_comparison.png')
    with tab4:
        st.image('../04_results/visualizations/05_complete_comparison.png')
except:
    st.info("Run compare_all_results.py to generate comparison plots")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>Crystallization Component Prediction System</b></p>
    <p>Advanced Baseline achieves: 64% Name | 47% Conc R² | 99% pH R²</p>
    <p>Built with Scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)
