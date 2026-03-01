import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go # Required for Sine Wave

# 1. Page Configuration
st.set_page_config(page_title="Advanced Smart Grid Monitor", layout="wide")

# Initialize Session State for Temporal Buffering
if "theft_counter" not in st.session_state:
    st.session_state.theft_counter = 0

# Professional UI Styling
st.markdown("""
    <style>
    .metric-container { background: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .status-online { color: #10b981; font-weight: bold; }
    .status-offline { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. Secure Credential Retrieval
try:
    CHANNEL_ID = st.secrets["TS_CHANNEL_ID"]
    READ_API_KEY = st.secrets["TS_READ_API_KEY"]
except KeyError:
    st.error("Missing Secrets: Please add TS_CHANNEL_ID and TS_READ_API_KEY in Streamlit Cloud Settings.")
    st.stop()

# 3. Resource Loading
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    cnn_model = tf.keras.models.load_model('combined_model.keras')
    iso_model = joblib.load('iso_forest.pkl')
    return scaler, cnn_model, iso_model

scaler, model, iso_forest = load_assets()

# 4. Data Fetching Logic
def fetch_live_data(results=60):
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={results}"
    try:
        response = requests.get(url, timeout=5).json()
        df = pd.DataFrame(response['feeds'])
        
        rename_map = {
            'field1': 'Voltage', 'field2': 'Current', 'field3': 'Power',
            'field4': 'Energy', 'field5': 'Frequency', 'field6': 'Power_Factor'
        }
        df.rename(columns=rename_map, inplace=True)
        df['Time'] = pd.to_datetime(df['created_at'])
        
        cols = ['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']
        df[cols] = df[cols].apply(pd.to_numeric)
        return df[['Time'] + cols]
    except Exception:
        return None

# --- Dashboard UI ---
st.title("🛡️ Pro-AI Electricity Theft Detection System")

with st.sidebar:
    st.header("⚙️ Settings")
    refresh_rate = st.slider("Data Refresh Rate (sec)", 16, 60, 20)
    
    if st.button("🗑️ Clear Alert History"):
        st.session_state.theft_counter = 0
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    st.info(f"Connected to Channel: {CHANNEL_ID}")

df = fetch_live_data()

if df is not None and not df.empty:
    latest = df.iloc[-1]
    
    # 5. ESP32 Connection Heartbeat
    last_update_time = latest['Time'].replace(tzinfo=None)
    is_online = (datetime.utcnow() - last_update_time) < timedelta(minutes=2)
    
    if is_online:
        st.markdown("### Status: <span class='status-online'>● ONLINE (ESP32 Connected)</span>", unsafe_allow_html=True)
    else:
        st.markdown("### Status: <span class='status-offline'>○ OFFLINE (Check ESP32 Power)</span>", unsafe_allow_html=True)

    # 6. Parameter Display Grid
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Voltage", f"{latest['Voltage']:.1f} V")
    m2.metric("Current", f"{latest['Current']:.3f} A")
    m3.metric("Power", f"{latest['Power']:.1f} W")
    m4.metric("Energy", f"{latest['Energy']:.2f} kWh")
    m5.metric("Frequency", f"{latest['Frequency']:.1f} Hz")
    m6.metric("PF", f"{latest['Power_Factor']:.2f}")

    st.divider()

    # 7. AI Analysis & Multi-Metric Device Intelligence
    col_ai, col_graph = st.columns([1, 2])

    with col_ai:
        st.subheader("🕵️ AI Intelligence")
        
        # Pull metrics
        p = latest['Power']
        pf = latest['Power_Factor']
        i = latest['Current']
        f = latest['Frequency']
        
        # Multi-Metric Recognition logic
        if i < 0.05:
            active_device = "Standby Mode"
        elif pf > 0.95:
            active_device = "40W Light" if 30 <= p < 45 else "High Resistive Load"
        elif pf < 0.70:
            active_device = "Laptop" if 25 <= p < 60 else "Mobile Phone Charging"
        elif 0.70 <= pf <= 0.95:
            active_device = "Mixed (Light/Laptop/Phone)"
        else:
            active_device = "Unidentified Combination"
        
        st.write(f"**Identified Device:** {active_device}")

        # AI Prediction
        raw_input = latest[['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']].values.reshape(1, -1)
        scaled_input = scaler.transform(raw_input)
        
        iso_status = iso_forest.predict(scaled_input)[0] 
        sequence = np.repeat(scaled_input[:, np.newaxis, :], 10, axis=1) 
        theft_prob = model.predict(sequence, verbose=0)[0][0]

        # 8. Optimized Detection Logic (Temporal Buffering)
        if theft_prob > 0.85:
            st.session_state.theft_counter += 1
        else:
            st.session_state.theft_counter = 0

        if st.session_state.theft_counter >= 3:
            st.error(f"🚨 CONFIRMED THEFT DETECTED ({theft_prob:.1%})")
            st.warning("Continuous anomalous load signature confirmed.")
        elif theft_prob > 0.85:
            st.info(f"🟡 Analyzing Pattern... ({st.session_state.theft_counter}/3)")
        elif iso_status == -1:
            st.warning("⚠️ Unknown Signature (Anomaly Found)")
        else:
            st.success("✅ System Secure: Signature Verified")

    with col_graph:
        st.subheader("⚡ Live AC Current Sine Wave")
        
        # --- GENERATE MATHEMATICAL SINE WAVE ---
        # t = time in seconds (0.04s = 2 cycles at 50Hz)
        t_sine = np.linspace(0, 0.04, 500)
        # y = Amplitude * sin(2 * pi * frequency * t)
        # We use the real-time frequency (f) and current (i) from sensor
        actual_f = f if f > 40 else 50.0 
        y_sine = i * np.sin(2 * np.pi * actual_f * t_sine)
        
        fig_sine = go.Figure()
        fig_sine.add_trace(go.Scatter(
            x=t_sine, y=y_sine,
            mode='lines',
            line=dict(color='#10b981', width=3),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))

        fig_sine.update_layout(
            height=300,
            xaxis_title="Time (seconds)",
            yaxis_title="Current (Amperes)",
            template="plotly_white",
            yaxis=dict(range=[-max(i*1.5, 0.5), max(i*1.5, 0.5)]), # Dynamic scaling
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_sine, use_container_width=True)

    # Secondary History Graph
    st.divider()
    st.subheader("📈 Power Consumption History (Watts)")
    st.line_chart(df.set_index('Time')['Power'])

# Auto-Refresh Logic
time.sleep(refresh_rate)
st.rerun()

