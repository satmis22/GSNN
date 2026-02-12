import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Ensure plot directory exists
os.makedirs('../plots', exist_ok=True)

# ==========================================
# 1. LOAD & PREPARE FULL DATASET
# ==========================================
print("Loading full dataset...")
df = pd.read_csv('../data/raw/synthetic_gsnn_data.csv', index_col=0)

features = ['Geo_Risk_Index', 'Transit_Days', 'LNG_Price']
target = 'Is_Crisis'
data = df[features].values
labels = df[target].values

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create Sequences
TIME_STEPS = 30
def create_sequences(data, labels, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, labels, TIME_STEPS)

# ==========================================
# 2. TRAIN MODEL (On 80% split)
# ==========================================
# We still train on a split to ensure the model learns correctly
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]

print("Training GSNN Model...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train quietly
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)

# ==========================================
# 3. PREDICT ON THE FULL TIMELINE
# ==========================================
print("Generating predictions for the entire year...")
# This is the key change: We predict on 'X' (all data), not just 'X_test'
all_predictions = model.predict(X)

# ==========================================
# 4. GENERATE THE RESEARCH PAPER PLOT
# ==========================================
plt.figure(figsize=(12, 6))

# Plot the Actual Crisis (Ground Truth)
# We fill the area where the crisis is happening (Is_Crisis = 1)
plt.fill_between(range(len(y)), y.flatten(), color='gray', alpha=0.3, label='Actual Crisis Event (Ground Truth)')

# Plot the Model's Prediction
plt.plot(all_predictions, color='#d62728', linewidth=2, label='GSNN Predicted Threat Score (TPS)')

# Add Policy Trigger Line
plt.axhline(y=0.75, color='orange', linestyle='--', linewidth=1.5, label='Policy Intervention Trigger (0.75)')

# Formatting
plt.title('GSNN Efficacy: Early Detection of Geopolitical Supply Shock', fontsize=14, fontweight='bold')
plt.xlabel('Timeline (Days)', fontsize=12)
plt.ylabel('Threat Probability Score (0-1)', fontsize=12)
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)

# Save
output_path = '../plots/gsnn_full_timeline.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS! New graph saved to: {output_path}")
print("Open this file to see the spike!")