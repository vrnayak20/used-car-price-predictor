import pandas as pd

# Load the dataset
df = pd.read_csv('data/cleaned_vehicles.csv')

cols = ['fuel_clean', 'drive_clean', 'transmission_clean', 'engine_clean']

print("--- VOCABULARY CHECK ---")
for col in cols:
    if col in df.columns:
        # Get unique values, convert to string, lower case, sort
        values = sorted(df[col].dropna().astype(str).str.lower().unique())
        print(f"\n[{col.upper()}] (What the model knows):")
        print(values)
    else:
        print(f"\n[{col.upper()}] - Missing from CSV")