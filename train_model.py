import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration
INPUT_PATH = 'data/large_vehicles.csv'
MODEL_PATH = 'models/price_model.joblib'

# --- 1. The Cleaning Logic (The "Universal Translator") ---
def standardize_fuel(val):
    if pd.isna(val): return 'unknown'
    val = str(val).lower().strip()
    if 'electric' in val: return 'electric'
    if 'hybrid' in val: return 'hybrid'
    if 'diesel' in val: return 'diesel'
    if 'gas' in val or 'premium' in val or 'unleaded' in val or 'flex' in val or 'e85' in val:
        return 'gas'
    return 'other'

def standardize_drive(val):
    if pd.isna(val): return 'unknown'
    val = str(val).lower().strip()
    if '4wd' in val or 'four' in val or '4x4' in val: return '4wd'
    if 'awd' in val or 'all' in val: return 'awd'
    if 'fwd' in val or 'front' in val: return 'fwd'
    if 'rwd' in val or 'rear' in val: return 'rwd'
    if '2wd' in val or '4x2' in val: return '2wd'
    return 'unknown'

def standardize_transmission(val):
    if pd.isna(val): return 'unknown'
    val = str(val).lower().strip()
    if 'manual' in val or 'm/t' in val or 'stick' in val:
        if 'auto' not in val: return 'manual'
    if 'auto' in val or 'a/t' in val or 'cvt' in val or 'pkd' in val or 'dsg' in val:
        return 'automatic'
    return 'other'

def standardize_engine(val):
    if pd.isna(val): return 'unknown'
    val = str(val).lower().strip()
    if 'electric' in val or 'motor' in val: return 'electric'
    if 'v8' in val or '8 cylinder' in val or '8 cyl' in val or '5.0l' in val or '5.3l' in val or '5.7l' in val or '6.2l' in val: return 'v8'
    if 'v6' in val or '6 cylinder' in val or '6 cyl' in val or '3.5l' in val or '3.6l' in val: return 'v6'
    if '4 cylinder' in val or '4 cyl' in val or 'i-4' in val or 'i4' in val or '2.4l' in val or '2.5l' in val or '1.5l' in val: return '4cyl'
    if 'diesel' in val: return 'diesel'
    return 'other'

# ---------------------------------------------------------

def train_model():
    print("Starting Training Pipeline...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_PATH)
        print(f"   - Loaded {len(df):,} rows.")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_PATH}.")
        return

    # 2. Basic Filtering & Renaming
    # Standardize 'odometer' to 'mileage' first thing
    if 'odometer' in df.columns:
        df = df.rename(columns={'odometer': 'mileage'})
    
    df = df.drop_duplicates()
    df = df.dropna(subset=['price', 'year', 'mileage'])

    # Strict Quality Control
    df = df[df['price'] > 1000] # Kept your logic, but 10000 might be too high? Adjusted to 1000 for safety.
    df = df[df['price'] < 100000]
    df = df[df['year'] > 2005]
    df = df[df['mileage'] < 250000]
    
    print(f"   - Filtered to {len(df):,} high-quality rows.")

    # --- 3. APPLY STANDARDIZATION ---
    print("   - Standardizing columns...")
    
    # We apply the logic directly to the Kaggle Column Names
    # If the column is missing, we fill with 'unknown' first
    
    # FUEL
    if 'fuel_type' in df.columns:
        df['fuel_clean'] = df['fuel_type'].apply(standardize_fuel)
    else:
        df['fuel_clean'] = 'unknown'

    # DRIVE
    if 'drivetrain' in df.columns:
        df['drive_clean'] = df['drivetrain'].apply(standardize_drive)
    else:
        df['drive_clean'] = 'unknown'

    # TRANSMISSION
    if 'transmission' in df.columns:
        df['transmission_clean'] = df['transmission'].apply(standardize_transmission)
    else:
        df['transmission_clean'] = 'unknown'

    # ENGINE (Sometimes 'engine' or 'cylinders')
    if 'engine' in df.columns:
        df['engine_clean'] = df['engine'].apply(standardize_engine)
    elif 'cylinders' in df.columns:
        df['engine_clean'] = df['cylinders'].apply(standardize_engine)
    else:
        df['engine_clean'] = 'unknown'
    
    # --- 4. Define Features ---
    # These are the Exact Column Names we want in X
    features = [
        'manufacturer', 'model', 'year', 'mileage', 
        'transmission_clean', 'fuel_clean', 'drive_clean', 'engine_clean'
    ]
    
    # Fill NA for the base columns
    for col in ['manufacturer', 'model']:
        df[col] = df[col].fillna('unknown').astype(str).str.lower()

    X = df[features]
    y = np.log1p(df['price'])

    # Debug: Show distribution
    print("\n   [Data Distribution after Cleaning]")
    print(f"   Fuel Types: {df['fuel_clean'].unique()}")
    print(f"   Drive Types: {df['drive_clean'].unique()}")
    print(f"   Engines: {df['engine_clean'].unique()}")
    print(f"   Transmissions: {df['transmission_clean'].unique()}")

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 6. Preprocessing
    # FIX: Ensure these match the 'features' list exactly
    categorical_cols = ['manufacturer', 'model', 'transmission_clean', 'fuel_clean', 'drive_clean', 'engine_clean']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), ['year', 'mileage']),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=100, sparse_output=False), 
             categorical_cols)
        ])

    # 7. Model (XGBoost)
    model = XGBRegressor(
        n_estimators=1500,       
        learning_rate=0.05,      
        max_depth=10,             
        subsample=0.7,           
        colsample_bytree=0.7,
        n_jobs=-1,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 8. Train
    print("\nTraining Model...")
    pipeline.fit(X_train, y_train)

    # 9. Evaluate
    print("Evaluating...")
    
    test_preds = np.expm1(pipeline.predict(X_test))
    train_preds = np.expm1(pipeline.predict(X_train))
    y_test_dollar = np.expm1(y_test)
    y_train_dollar = np.expm1(y_train)
    
    test_mae = mean_absolute_error(y_test_dollar, test_preds)
    train_mae = mean_absolute_error(y_train_dollar, train_preds)
    test_r2 = r2_score(y_test_dollar, test_preds)

    print(f"\nModel Results:")
    print(f"   - Testing MAE:  ${test_mae:,.0f}")
    print(f"   - Training MAE: ${train_mae:,.0f}")
    print(f"   - Testing R^2:  {test_r2:.4f}")
    print(f"   - Overfitting Gap: ${test_mae - train_mae:,.0f}")

    # 10. Save
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()