import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import time
from datetime import datetime, timedelta

# 1. Page & Security Configuration
st.set_page_config(page_title="Pro-Smart Grid Monitor", layout="wide")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    .metric-container { background: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .status-online { color: #10b981; font-weight: bold; }
    .status-offline { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# CORRECT WAY: Use the Variable Names, not the values
try:
    CHANNEL_ID = st.secrets["TS_CHANNEL_ID"]
    READ_API_KEY = st.secrets["TS_READ_API_KEY"]
except:
    st.error("Missing Secrets: Ensure TS_CHANNEL_ID and TS_READ_API_KEY are in Streamlit Cloud Settings.")
    st.stop()

# 2. Load Unified AI Assets
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    cnn_model = tf.keras.models.load_model('combined_model.keras')
    iso_model = joblib.load('iso_forest.pkl')
    return scaler, cnn_model, iso_model

scaler, model, iso_forest = load_assets()

# 3. Enhanced Data Fetching
def fetch_live_data(results=60):
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={results}"
    try:
        response = requests.get(url, timeout=5).json()
        df = pd.DataFrame(response['feeds'])
        
        # Professional Mapping
        rename_map = {
            'field1': 'Voltage', 'field2': 'Current', 'field3': 'Power',
            'field4': 'Energy', 'field5': 'Frequency', 'field6': 'Power_Factor'
        }
        df.rename(columns=rename_map, inplace=True)
        df['Time'] = pd.to_datetime(df['created_at'])
        
        cols = ['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']
        df[cols] = df[cols].apply(pd.to_numeric)
        return df[['Time'] + cols]
    except:
        return None

# --- Dashboard Layout ---
st.title("🛡️ Advanced Smart Meter Security Dashboard")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    refresh_rate = st.slider("Refresh Interval (sec)", 16, 60, 16)
    
    if st.button("🗑️ Clear Local History"):
        st.cache_data.clear()
        st.success("Cache Cleared")
    
    st.divider()
    st.info(f"Monitoring Channel: {CHANNEL_ID}")

# Fetch Current State
df = fetch_live_data()

if df is not None and not df.empty:
    latest = df.iloc[-1]
    
    # 4. Connection Status Logic (Heartbeat)
    last_update_time = latest['Time'].replace(tzinfo=None)
    is_online = (datetime.utcnow() - last_update_time) < timedelta(minutes=2)
    
    status_text = "🟢 ESP32: ONLINE" if is_online else "🔴 ESP32: OFFLINE"
    st.subheader(status_text)

    # 5. Live Parameters (All Parameters)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Voltage", f"{latest['Voltage']:.1f}V")
    m2.metric("Current", f"{latest['Current']:.3f}A")
    m3.metric("Power", f"{latest['Power']:.1f}W")
    m4.metric("Energy", f"{latest['Energy']:.2f}kWh")
    m5.metric("Frequency", f"{latest['Frequency']:.1f}Hz")
    m6.metric("PF", f"{latest['Power_Factor']:.2f}")

    st.divider()

    # 6. Device Recognition & AI Analysis
    col_ai, col_graph = st.columns([1, 2])

    with col_ai:
        st.subheader("🕵️ Device Intelligence")
        
        # Prepare for AI
        raw_input = latest[['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']].values.reshape(1, -1)
        scaled_input = scaler.transform(raw_input)
        
        # A. Identity Recognition (Based on Power Levels)
        p = latest['Power']
        if p < 5: active_device = "Standby / Idle"
        elif 5 <= p < 25: active_device = "Phone Charging"
        elif 25 <= p < 50: active_device = "40W Light"
        elif 50 <= p < 100: active_device = "Laptop"
        else: active_device = "Multiple / Heavy Load"
        
        st.write(f"**Current Device:** {active_device}")

        # B. AI Anomaly/Theft Check
        iso_status = iso_forest.predict(scaled_input)[0]
        sequence = np.repeat(scaled_input[:, np.newaxis, :], 10, axis=1)
        theft_prob = model.predict(sequence)[0][0]

        # New code (Make it less sensitive):
        if theft_prob > 0.85: # Increased threshold to reduce false alarms
            st.error("🚨 ALERT: Theft Detected")
        elif iso_status == -1:
            st.warning("⚠️ Unknown Signature (Anomalous Device)")
        else:
            st.success("✅ Signature Verified: Secure")

    with col_graph:
        st.subheader("📈 Real-Time Power Load")
        st.line_chart(df.set_index('Time')['Power'])

# Auto-Refresh Logic
time.sleep(refresh_rate)
st.rerun()


