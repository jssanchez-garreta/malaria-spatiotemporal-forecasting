# ============================================================
# LSTM REGRESSOR - INDEPENDENT PIPELINE
# ============================================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. LOAD DATA & PREPARE SEQUENCES
# ------------------------------------------------------------
file_name = "final_dataset.csv"

if not os.path.exists(file_name):
    from google.colab import files
    print("Por favor, sube el archivo final_dataset.csv:")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]

df = pd.read_csv(file_name)
df = df.sort_values(by=["country_region", "year"]).reset_index(drop=True)
df["precip_lag1"] = df.groupby("country_region")["precipitation"].shift(1)
df = df.dropna().reset_index(drop=True)

features = ["temperature", "precipitation", "humidity", "precip_lag1", "incidence"]

scaler = StandardScaler()
train_mask_scaler = df["year"] <= 2016
scaler.fit(df.loc[train_mask_scaler, features])

df_scaled = df.copy()
df_scaled[features] = scaler.transform(df[features])

TIME_STEPS = 5
X, y, y_years, y_regions = [], [], [], []

for region in df_scaled["country_region"].unique():
    region_df = df_scaled[df_scaled["country_region"] == region].sort_values("year")
    vals = region_df[features].values
    yrs = region_df["year"].values

    for i in range(len(vals) - TIME_STEPS):
        X.append(vals[i:i+TIME_STEPS])
        y.append(vals[i+TIME_STEPS, -1])
        y_years.append(yrs[i+TIME_STEPS])
        y_regions.append(region)

X, y = np.array(X), np.array(y)
y_years, y_regions = np.array(y_years), np.array(y_regions)

train_mask = y_years <= 2016
val_mask   = (y_years >= 2017) & (y_years <= 2020)
test_mask  = y_years >= 2021

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[val_mask], y[val_mask]
X_test, y_test   = X[test_mask], y[test_mask]

# ------------------------------------------------------------
# 2. MODEL TRAINING & PREDICTION
# ------------------------------------------------------------
print("\n--- Entrenando LSTM ---")
model = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

y_pred_norm = model.predict(X_test, verbose=0).flatten()

# Desnormalización
mean_inc, std_inc = scaler.mean_[-1], scaler.scale_[-1]
y_test_real = y_test * std_inc + mean_inc
y_pred_real = y_pred_norm * std_inc + mean_inc

mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2   = r2_score(y_test_real, y_pred_real)

print(f"\n📊 Resultados LSTM -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

# ------------------------------------------------------------
# 3. EXPORT SUMMARY & PREDICTIONS
# ------------------------------------------------------------
summary_df = pd.DataFrame([{"Model": "LSTM", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}])
summary_df.to_excel("lstm_summary.xlsx", index=False)
summary_df.to_csv("lstm_summary.csv", index=False)

pred_df = pd.DataFrame({
    "country_region": y_regions[test_mask],
    "year": y_years[test_mask],
    "y_true": y_test_real,
    "y_pred": y_pred_real
})
pred_df.to_csv("predictions_lstm.csv", index=False)

print("\n✅ Archivos guardados: lstm_summary.xlsx, lstm_summary.csv, predictions_lstm.csv")