
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Premium Soft Wine & Charcoal Elegant Aesthetic
st.set_page_config(page_title="Fraud Sentry Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Main App Background - Elegant Charcoal Dark */
    .stApp { 
        background: radial-gradient(circle, #250206 0%, #120102 100%);
        color: #F8F9FA; 
    }
    
    /* Elegant Title and Header styling */
    h1 { 
        color: #E5A93C !important; 
        font-family: 'Georgia', serif; 
        font-size: 40px; 
        text-align: center; 
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    /* Card Container for Inputs and Results */
    .custom-card {
        background-color: #1C0306;
        border: 1px solid rgba(229, 169, 60, 0.2);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    
    /* Input field text colors */
    label, .stMarkdown p { 
        color: #E0D4D5 !important; 
        font-size: 15px; 
        font-weight: 500;
    }
    
    /* Clean Input Box Styling */
    .stNumberInput input, .stSelectbox div, .stSlider div { 
        background-color: #0E0102 !important; 
        color: #FFFFFF !important; 
        border: 1px solid rgba(229, 169, 60, 0.3) !important; 
        border-radius: 8px !important;
    }
    
    /* Velvet Action Button */
    .stButton>button { 
        background: linear-gradient(135deg, #990011 0%, #55000A 100%); 
        color: #FFFFFF !important; 
        border: 1px solid #E5A93C; 
        border-radius: 8px; 
        width: 100%; 
        height: 52px; 
        font-size: 18px; 
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(153, 0, 17, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(229, 169, 60, 0.4);
        border-color: #FFFFFF;
    }
    
    /* Soft Status Output Boxes */
    .result-box-safe { 
        background: linear-gradient(135deg, rgba(20, 50, 20, 0.6) 0%, rgba(10, 30, 10, 0.8) 100%);
        border: 2px solid #2ECC71; 
        padding: 30px; 
        border-radius: 12px; 
        text-align: center; 
    }
    .result-box-fraud { 
        background: linear-gradient(135deg, rgba(80, 10, 15, 0.6) 0%, rgba(40, 5, 5, 0.8) 100%);
        border: 2px solid #E74C3C; 
        padding: 30px; 
        border-radius: 12px; 
        text-align: center; 
    }
    
    /* Asset error beautifully customized */
    .stAlert {
        background-color: #3D080E !important;
        color: #FFC107 !important;
        border: 1px solid #E5A93C !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>👑 FRAUD SENTRY DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #D2C4C5; font-size: 16px; margin-bottom: 25px;'>Advanced Real-Time Financial Risk Intelligence System</p>", unsafe_allow_html=True)

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
    st.error("⚠️ System Notice: 'knn_model.pkl' or 'scaler.pkl' not found locally. Please ensure files are committed to your GitHub repository root folder.")

# 3. Clean Spaced Layout
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #E5A93C; margin-top:0;'>📝 Transaction Metrics</h3>", unsafe_allow_html=True)
    
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
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #E5A93C; margin-top:0;'>📊 Operational Integrity Report</h3>", unsafe_allow_html=True)
    
    if predict_btn:
        try:
            scaled_input = scaler.transform([input_data])
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]
            
            if prediction == 1:
                st.markdown(f"""
                    <div class='result-box-fraud'>
                        <h2 style='color: #E74C3C; margin: 0; font-size: 24px;'>🚨 TRANSACTION BLOCKED</h2>
                        <p style='color: #FFFFFF; font-size: 16px; margin-top: 10px;'>High Probability Fraud Pattern Detected</p>
                        <h4 style='color: #E5A93C; margin: 5px;'>Confidence: {prediction_proba[1]*100:.1f}%</h4>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='result-box-safe'>
                        <h2 style='color: #2ECC71; margin: 0; font-size: 24px;'>✅ TRANSACTION CLEAN</h2>
                        <p style='color: #FFFFFF; font-size: 16px; margin-top: 10px;'>Authorized and Cleared for Settlement</p>
                        <h4 style='color: #E5A93C; margin: 5px;'>Confidence: {prediction_proba[0]*100:.1f}%</h4>
                    </div>
                """, unsafe_allow_html=True)
        except NameError:
            st.warning("Cannot run evaluation because model/scaler assets are not loaded.")
    else:
        st.info("System Ready. Awaiting parameter submission from the left panel.")
    st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #E5A93C; margin-top:0;'>📈 Core Decision Drivers</h3>", unsafe_allow_html=True)
    st.caption("Operational feature impact weights computed by Mutual Information ranking:")
    
    # Beautiful Matplotlib chart adjustment
    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor('#1C0306')
    ax.set_facecolor('#0E0102')
    
    features = ['Transaction Hour', 'Shipping Distance', 'Avg Amount', 'Account Age', 'Amount']
    importance = [0.008, 0.024, 0.026, 0.029, 0.033]
    
    ax.barh(features, importance, color='#990011', edgecolor='#E5A93C', height=0.55)
    ax.tick_params(colors='#F8F9FA', labelsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.15, color='#F8F9FA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('rgba(229, 169, 60, 0.4)')
    ax.spines['bottom'].set_color('rgba(229, 169, 60, 0.4)')
    
    import streamlit as st
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
