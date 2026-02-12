import pandas as pd
import numpy as np
import os

# ==========================================
# 1. ROBUST PATH SETUP (The Fix)
# ==========================================
# Get the folder where THIS script lives (GSNN/src)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the Project Root (GSNN/) by going up one level
project_root = os.path.dirname(script_dir)

# Define exact path for data output (GSNN/data/raw)
data_dir = os.path.join(project_root, 'data', 'raw')

# Create the folder if it doesn't exist (This fixes the "Folder not found" error)
os.makedirs(data_dir, exist_ok=True)

print(f"Target Data Directory: {data_dir}")

# ==========================================
# 2. GENERATE DATA
# ==========================================
np.random.seed(42)
days = 365
date_range = pd.date_range(start='2025-01-01', periods=days, freq='D')

# Initialize DataFrame
df = pd.DataFrame(index=date_range)

# Features
df['Geo_Risk_Index'] = np.random.uniform(0.1, 0.2, days)
df['Transit_Days'] = np.random.normal(18, 1.5, days)
df['LNG_Price'] = np.random.normal(12, 0.5, days)
df['Is_Crisis'] = 0

# Inject Crisis (Days 200-260)
start_event = 200
end_event = 260

# Spikes
df.iloc[start_event:end_event, 0] = np.random.uniform(0.8, 0.95, end_event-start_event) # Geo Risk
df.iloc[start_event+5:end_event+5, 1] += np.random.normal(12, 2, end_event-start_event) # Transit Lag
df.iloc[start_event+2:end_event+2, 2] += np.random.normal(8, 1.5, end_event-start_event) # Price Lag
df.iloc[start_event:end_event, 3] = 1 # Label

# ==========================================
# 3. SAVE DATA
# ==========================================
# Define the full file path
output_path = os.path.join(data_dir, 'synthetic_gsnn_data.csv')

# Save to CSV
df.to_csv(output_path)

print(f"\nSUCCESS! Data generated and saved to:")
print(f" -> {output_path}")
print("\nFirst 5 rows:")
print(df.head())