import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Premium Dark Wine & Velvet Aesthetic Setup
st.set_page_config(page_title="Fraud Sentry Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Main Background: Deep Wine/Burgundy Pigment */
    .stApp { background-color: #2D0202; color: #F5F5DC; }
    
    /* Headings & Text Colors */
    h1 { color: #D4AF37 !important; font-family: 'Playfair Display', serif; font-size: 42px; text-align: center; font-weight: 700; }
    h3 { color: #D4AF37 !important; font-family: 'Playfair Display', serif; }
    p, label, .stMarkdown { color: #F5F5DC !important; font-size: 16px; }
    
    /* Input Boxes Custom Design */
    .stNumberInput input, .stSelectbox div, .stSlider div { 
        background-color: #1a0101 !important; 
        color: #F5F5DC !important; 
        border: 1px solid rgba(212, 175, 55, 0.4) !important; 
    }
    
    /* Graduated Action Button */
    .stButton>button { 
        background: linear-gradient(135deg, #8B0000 0%, #4A0404 100%); 
        color: #D4AF37 !important; 
        border: 1px solid #D4AF37; 
        border-radius: 8px; width: 100%; height: 50px; font-size: 18px; font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); border-color: #FFFFFF; }
    
    /* Status Boxes */
    .result-box-safe { background-color: rgba(0, 128, 0, 0.25); border: 2px solid #00FF00; padding: 25px; border-radius: 12px; text-align: center; }
    .result-box-fraud { background-color: rgba(139, 0, 0, 0.45); border: 2px solid #FF3333; padding: 25px; border-radius: 12px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🍷 FRAUD SENTRY: FINANCIAL RISK DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #D4AF37; font-size: 18px;'>Graduated KNN Engine & Real-Time Operational Insights</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid rgba(212, 175, 55, 0.2);'>", unsafe_allow_html=True)

# 2. Asset Loader
@st.cache_resource
def load_assets():
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Assets Error: Make sure 'knn_model.pkl' and 'scaler.pkl' are uploaded to GitHub.")

# 3. Two-Column Dashboard Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📝 Input Transaction Metrics")
    
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)
    shipping_distance_km = st.number_input("Shipping Distance (KM)", min_value=0.0, value=45.0)
    avg_amount_user = st.number_input("User Average Historical Amount ($)", min_value=0.0, value=120.0)
    total_transactions_user = st.number_input("Total Historical Transactions", min_value=0, value=25)
    transaction_hour = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
    
    avs_match = st.selectbox("AVS Match Status (Address Verified)", [1, 0])
    cvv_result = st.selectbox("CVV Verification Result", [1, 0])
    three_ds_flag = st.selectbox("3D Secure Dynamic Flag", [1, 0])
    promo_used = st.selectbox("Promo Code Applied", [0, 1])
    
    # 33 features placeholder array matching notebook
    input_data = np.zeros(33)
    input_data[0] = account_age_days
    input_data[1] = total_transactions_user
    input_data[2] = avg_amount_user
    input_data[3] = amount
    input_data[4] = promo_used
    input_data[5] = avs_match
    input_data[6] = cvv_result
    input_data[7] = three_ds_flag
    input_data[8] = shipping_distance_km
    input_data[9] = transaction_hour

    st.write("")
    predict_btn = st.button("🍷 Run Risk Evaluation")

with col2:
    st.markdown("### 📊 Operational Integrity Report")
    
    if predict_btn:
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        
        if prediction == 1:
            st.markdown(f"""
                <div class='result-box-fraud'>
                    <h2 style='color: #FF3333; margin: 0;'>🚨 TRANSACTION BLOCKED</h2>
                    <p style='color: #FFFFFF; font-size: 18px; margin-top: 10px;'>High Probability Fraud Pattern Detected</p>
                    <h3 style='color: #D4AF37; margin: 5px;'>Confidence: {prediction_proba[1]*100:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box-safe'>
                    <h2 style='color: #33FF33; margin: 0;'>✅ TRANSACTION CLEAN</h2>
                    <p style='color: #FFFFFF; font-size: 18px; margin-top: 10px;'>Authorized and Cleared for Settlement</p>
                    <h3 style='color: #D4AF37; margin: 5px;'>Confidence: {prediction_proba[0]*100:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Awaiting input execution. Adjust the system metrics on the left panel.")
            
    st.write("")
    st.markdown("### 📈 Core Decision Metrics (Non-Tech Explanation)")
    st.caption("This visualization breaks down the internal metric weights calculated for the model's structural features:")
    
    # Elegant custom colored feature importance chart
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#2D0202')
    ax.set_facecolor('#1a0101')
    
    features = ['Transaction Hour', 'Shipping Distance', 'Avg Amount', 'Account Age', 'Amount']
    importance = [0.008, 0.024, 0.026, 0.029, 0.033]
    
    ax.barh(features, importance, color='#8B0000', edgecolor='#D4AF37', height=0.5)
    ax.set_title("Top Operational Drivers Behind Risk Assessment", color='#D4AF37', fontsize=11, weight='bold')
    ax.tick_params(colors='#F5F5DC', labelsize=9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, color='#F5F5DC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
