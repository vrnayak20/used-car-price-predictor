from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import requests
import numpy as np

# --- Configuration ---
MODEL_PATH = 'models/price_model.joblib'
MARKET_ADJUSTMENT_SCALAR = 1.0

# --- Standardization Logic ---

def standardize_fuel(val):
    if pd.isna(val) or val is None: return 'unknown'
    val = str(val).lower().strip()
    if 'electric' in val: return 'electric'
    if 'hybrid' in val: return 'hybrid'
    if 'diesel' in val: return 'diesel'
    if 'gas' in val or 'premium' in val or 'unleaded' in val or 'flex' in val or 'e85' in val or 'regular' in val:
        return 'gas'
    return 'other'

def standardize_drive(val):
    if pd.isna(val) or val is None: return 'unknown'
    val = str(val).lower().strip()
    if '4wd' in val or 'four' in val or '4x4' in val: return '4wd'
    if 'awd' in val or 'all' in val: return 'awd'
    if 'fwd' in val or 'front' in val: return 'fwd'
    if 'rwd' in val or 'rear' in val: return 'rwd'
    if '2wd' in val or '4x2' in val: return '2wd'
    return 'unknown'

def standardize_transmission(val):
    if pd.isna(val) or val is None: return 'unknown'
    val = str(val).lower().strip()
    if 'manual' in val or 'm/t' in val or 'stick' in val:
        if 'auto' not in val: return 'manual'
    if 'auto' in val or 'a/t' in val or 'cvt' in val or 'pkd' in val or 'dsg' in val or 'continuously' in val:
        return 'automatic'
    return 'other'

def standardize_engine(val):
    if pd.isna(val) or val is None: return 'unknown'
    val = str(val).lower().strip()
    if 'electric' in val or 'motor' in val: return 'electric'
    if val == '8' or 'v8' in val or '8 cylinder' in val or '8 cyl' in val or '5.0l' in val or '5.3l' in val or '5.7l' in val or '6.2l' in val: return 'v8'
    if val == '6' or 'v6' in val or '6 cylinder' in val or '6 cyl' in val or '3.5l' in val or '3.6l' in val: return 'v6'
    if val == '4' or '4 cylinder' in val or '4 cyl' in val or 'i-4' in val or 'i4' in val or '2.4l' in val or '2.5l' in val or '1.5l' in val: return '4cyl'
    if 'diesel' in val: return 'diesel'
    return 'other'

# --- App Initialization ---
app = FastAPI(
    title="Car Price Arbitrage API",
    description="Predicts fair market value using XGBoost with Advanced Features",
    version="2.2"
)

# --- Load Model ---
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model from {MODEL_PATH}")
    model_pipeline = None

# --- Data Contracts ---
class VinInput(BaseModel):
    vin: str
    mileage: float
    year: int 

class ManualInput(BaseModel):
    brand: str
    model: str
    year: int
    mileage: float
    transmission: str = "unknown"
    fuel_type: str = "unknown"
    drivetrain: str = "unknown"
    cylinders: str = "unknown"

# --- Helper: VIN Decoder (UPDATED TO FLAT JSON) ---
def decode_vin_nhtsa(vin: str):
    """Call NHTSA Public API (Flat Values Endpoint)"""
    # NOTE: Changed endpoint to DecodeVinValues (returns flat JSON object)
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        results = data.get('Results', [])
        
        if not results:
            print("NHTSA API returned no results.")
            return None
            
        item = results[0] # DecodeVinValues returns a list with 1 item containing all fields
        
        # DEBUG LOGGING (Check your terminal if this fails!)
        print(f"\n--- DEBUG VIN: {vin} ---")
        print(f"Raw Fuel: {item.get('FuelTypePrimary')}")
        print(f"Raw Drive: {item.get('DriveType')}")
        print(f"Raw Cylinders: {item.get('EngineCylinders')}")
        print(f"Raw Trans: {item.get('TransmissionStyle')}")
        print("------------------------\n")

        details = {
            'make': item.get('Make'),
            'model': item.get('Model'),
            'fuel': item.get('FuelTypePrimary'),
            'drive': item.get('DriveType'),
            'cylinders': item.get('EngineCylinders'),
            'transmission': item.get('TransmissionStyle')
        }
        
        # Ensure we don't pass empty strings
        for key, val in details.items():
            if val == "": details[key] = "unknown"
                
        if details['make'] and details['model']:
            return details
        return None
    except Exception as e:
        print(f"API Connection Error: {e}")
        return None

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "healthy"}

@app.post("/predict/manual")
def predict_manual(car: ManualInput):
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Create DataFrame
    input_data = {
        'manufacturer': [car.brand.lower().strip()],
        'model': [car.model.lower().strip()],
        'year': [car.year],
        'mileage': [car.mileage],
    }
    df = pd.DataFrame(input_data)

    # 2. Standardization
    df['fuel_clean'] = standardize_fuel(car.fuel_type)
    df['drive_clean'] = standardize_drive(car.drivetrain)
    df['transmission_clean'] = standardize_transmission(car.transmission)
    df['engine_clean'] = standardize_engine(car.cylinders)

    # 3. Predict
    try:
        log_price = model_pipeline.predict(df)[0]
        predicted_price_2020 = float(np.expm1(log_price))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model error: {str(e)}")

    # 4. Range
    MAE_MARGIN = 2500
    low_bound = predicted_price_2020 - MAE_MARGIN
    high_bound = predicted_price_2020 + MAE_MARGIN
    
    final_price = predicted_price_2020 * MARKET_ADJUSTMENT_SCALAR
    final_low = low_bound * MARKET_ADJUSTMENT_SCALAR
    final_high = high_bound * MARKET_ADJUSTMENT_SCALAR

    return {
        "input": car,
        "predicted_price": round(final_price, 2),
        "confidence_interval": {
            "low": round(final_low, 2),
            "high": round(final_high, 2)
        },
        "details": {
            "base_prediction_2020": round(predicted_price_2020, 2),
            "features_detected": {
                "engine": df['engine_clean'][0],
                "drive": df['drive_clean'][0],
                "fuel": df['fuel_clean'][0],
                "transmission": df['transmission_clean'][0]
            }
        }
    }

@app.post("/predict/vin")
def predict_vin(vin_data: VinInput):
    # 1. Decode
    details = decode_vin_nhtsa(vin_data.vin)
    
    if not details:
        raise HTTPException(status_code=400, detail="Could not decode VIN.")
    
    # 2. Forward
    car_input = ManualInput(
        brand=details['make'],
        model=details['model'],
        year=vin_data.year,
        mileage=vin_data.mileage,
        fuel_type=details['fuel'] or "unknown",
        drivetrain=details['drive'] or "unknown",
        cylinders=details['cylinders'] or "unknown",
        transmission=details['transmission'] or "unknown"
    )
    
    response = predict_manual(car_input)
    response['vin_decoded'] = True
    response['vin'] = vin_data.vin
    return response