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
# 1. LOAD DATA
# ==========================================
print("Loading data...")
df = pd.read_csv('../data/raw/synthetic_gsnn_data.csv', index_col=0)

features = ['Geo_Risk_Index', 'Transit_Days', 'LNG_Price']
target = 'Is_Crisis'
data = df[features].values
labels = df[target].values

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Sequence Creation
def create_sequences(data, labels, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

TIME_STEPS = 30
X, y = create_sequences(scaled_data, labels, TIME_STEPS)

# Split Train/Test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ==========================================
# 2. BUILD LSTM
# ==========================================
print("Building model...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. TRAIN
# ==========================================
print("Training model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# ==========================================
# 4. PREDICT & VISUALIZE
# ==========================================
print("Generating evaluation metrics...")
predictions = model.predict(X_test)

plt.figure(figsize=(14, 7))
plt.fill_between(range(len(y_test)), y_test.flatten(), color='gray', alpha=0.3, label='Actual Crisis Event')
plt.plot(predictions.flatten(), color='#d62728', linewidth=2.5, label='GSNN Predicted Threat Score')
plt.axhline(y=0.75, color='orange', linestyle='--', label='Policy Trigger (0.75)')

plt.title('GSNN Validation: Predicting Geopolitical Shock', fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.5)

output_path = '../plots/gsnn_results.png'
plt.savefig(output_path, dpi=300)
print(f"Success! Graph saved to: {output_path}")