import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

VEH_USAGE_OPTIONS = [
    "Private",
    "Private+trip to office",
    "Professional",
    "Professional run",
]

st.set_page_config(page_title="End-to-End Loss Cost Model", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #f8f9fa;
    }

    .main .block-container {
        padding-top: 3rem;
        max-width: 520px;
    }

    h1 {
        color: #0f172a;
        font-weight: 600;
        font-size: 1.75rem;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem;
    }

    label, .stNumberInput label, .stSelectbox label {
        color: #475569 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }

    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        color: #0f172a !important;
    }

    .stButton > button {
        background-color: #0f172a;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        font-size: 0.9rem;
        width: 100%;
        transition: background 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #1e293b;
        color: #ffffff;
    }

    .result {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1.75rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1.75rem;
    }
    .result-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 2.25rem;
        font-weight: 600;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .result-period {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("End-to-End Loss Cost Model")

with st.form("policy_form"):
    risk_area   = st.number_input("Risk Area", min_value=1.0, max_value=11.0, value=5.0, step=1.0)
    bonus_malus = st.number_input("Bonus-Malus", min_value=50, max_value=350, value=100, step=1)
    driv_age    = st.number_input("Driver Age", min_value=18, max_value=100, value=35, step=1)
    lic_age     = st.number_input("License Age (months)", min_value=0, max_value=900, value=120, step=1)
    veh_usage   = st.selectbox("Vehicle Usage", VEH_USAGE_OPTIONS, index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "RiskArea":   float(risk_area),
        "BonusMalus": int(bonus_malus),
        "DrivAge":    int(driv_age),
        "LicAge":     int(lic_age),
        "VehUsage":   veh_usage,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        pp = r.json()["loss_cost"]
        st.markdown(f"""
        <div class="result">
            <div class="result-label">Loss Cost</div>
            <div class="result-value">${pp:,.2f}</div>
            <div class="result-period">per year</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Request failed: {e}")
