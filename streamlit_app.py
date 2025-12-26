import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Car Price Arbitrage", layout="centered")

# --- Header ---
st.title("Used Car Price Arbitrage")
st.markdown("""
Predicts **Fair Market Value** using XGBoost on 700k+ records.
Features **VIN Decoding** to automatically detect Engine, Drive, and Fuel types.
""")
st.markdown("---")

# --- Tabs ---
tab1, tab2 = st.tabs(["VIN Search (Recommended)", "Manual Input"])

# --- TAB 1: VIN SEARCH ---
with tab1:
    st.header("Validate & Predict by VIN")
    st.info("Uses NHTSA Government API to detect specs (V8 vs V6, AWD vs FWD, etc) for higher accuracy.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        vin_input = st.text_input("Enter VIN (17 Characters)", max_chars=17, placeholder="e.g. 1HGCM...")
    with col2:
        vin_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2015)
    
    vin_mileage = st.number_input("Mileage", min_value=0, step=1000, value=118000, key="vin_mile")
    
    if st.button("Analyze VIN", type="primary"):
        if len(vin_input) != 17:
            st.error("VIN must be exactly 17 characters.")
        else:
            with st.spinner("Decoding VIN & Analyzing Market..."):
                payload = {
                    "vin": vin_input,
                    "mileage": vin_mileage,
                    "year": vin_year
                }
                try:
                    response = requests.post(f"{API_URL}/predict/vin", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.success("✅ VIN Decoded Successfully!")
                        
                        # --- Results Display ---
                        st.markdown(f"### 💰 Estimated Price: ${data['predicted_price']:,.2f}")
                        st.caption(f"Confidence Range: ${data['confidence_interval']['low']:,.2f} - ${data['confidence_interval']['high']:,.2f}")
                        
                        # Car Details Card
                        st.markdown("#### Vehicle Specs Detected")
                        st.markdown("The system automatically extracted these features from the VIN:")
                        
                        # Use columns for a clean look
                        specs = data['input']
                        features = data['details']['features_detected']
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Make/Model", f"{specs['brand']} {specs['model']}")
                        c2.metric("Engine", features['engine'].upper())
                        c3.metric("Transmission", features['transmission'].title())
                        
                        c4, c5, c6 = st.columns(3)
                        c4.metric("Drivetrain", features['drive'].upper())
                        c5.metric("Fuel", features['fuel'].title())
                        c6.metric("Year", specs['year'])
                        
                    else:
                        st.error(f"Server Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection Error: {e}. Is the backend running?")

# --- TAB 2: MANUAL INPUT ---
with tab2:
    st.header("Manual Estimator")
    
    col_a, col_b = st.columns(2)
    with col_a:
        brand = st.text_input("Brand", placeholder="ford").lower().strip()
    with col_b:
        model = st.text_input("Model", placeholder="f-150").lower().strip()
        
    col_c, col_d = st.columns(2)
    with col_c:
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2018, key="man_year")
    with col_d:
        mileage = st.number_input("Mileage", min_value=0, step=1000, value=50000, key="man_mile")
    
    st.markdown("---")
    st.markdown("**Advanced Specs (Optional)**")
    
    col_e, col_f = st.columns(2)
    with col_e:
        drive = st.selectbox("Drivetrain", ["unknown", "4wd", "awd", "fwd", "rwd"])
        fuel = st.selectbox("Fuel Type", ["unknown", "gas", "diesel", "hybrid", "electric"])
    with col_f:
        # We map these to the raw strings the API expects
        cylinders = st.selectbox("Cylinders", ["unknown", "4 cylinders", "6 cylinders", "8 cylinders", "other"])
        trans = st.selectbox("Transmission", ["unknown", "automatic", "manual"])
        
    if st.button("Predict Price (Manual)"):
        if not brand or not model:
            st.warning("Brand and Model are required.")
        else:
            payload = {
                "brand": brand,
                "model": model,
                "year": year,
                "mileage": mileage,
                "drivetrain": drive,
                "fuel_type": fuel,
                "cylinders": cylinders,
                "transmission": trans
            }
            try:
                response = requests.post(f"{API_URL}/predict/manual", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(f"### 💰 Estimated Price: ${data['predicted_price']:,.2f}")
                    st.caption(f"Range: ${data['confidence_interval']['low']:,.2f} - ${data['confidence_interval']['high']:,.2f}")
                    
                    # Debug Info
                    with st.expander("See Interpretation"):
                        st.write("Model Inputs Used:", data['details']['features_detected'])
                else:
                    st.error(f"Error: {response.json().get('detail')}")
            except Exception:
                st.error("Connection Error. Is the backend running?")