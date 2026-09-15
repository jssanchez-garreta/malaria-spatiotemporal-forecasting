# ============================================================
# GENERALIZED ADDITIVE MODEL (GAM) - INDEPENDENT PIPELINE
# ============================================================

import sys
import subprocess

# Auto-instalación de pygam si no está instalado en el entorno
try:
    from pygam import LinearGAM, s
except ModuleNotFoundError:
    print("📦 Instalando pygam...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygam"])
    from pygam import LinearGAM, s

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. LOAD DATA (WITH COLAB UPLOAD FALLBACK)
# ------------------------------------------------------------
file_name = "final_dataset.csv"

if not os.path.exists(file_name) and not os.path.exists("train_df.csv"):
    from google.colab import files
    print("Por favor, sube el archivo final_dataset.csv:")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]

if os.path.exists("train_df.csv"):
    train_df = pd.read_csv("train_df.csv")
    test_df  = pd.read_csv("test_df.csv")
else:
    df = pd.read_csv(file_name)
    df = df.sort_values(by=["country_region", "year"]).reset_index(drop=True)
    df["precip_lag1"] = df.groupby("country_region")["precipitation"].shift(1)
    df["incidence_lag1"] = df.groupby("country_region")["incidence"].shift(1)
    df = df.dropna().reset_index(drop=True)

    train_df = df[df["year"] <= 2016].copy()
    test_df  = df[df["year"] >= 2021].copy()

features = ["temperature", "precipitation", "humidity", "precip_lag1", "incidence_lag1"]

X_train, y_train = train_df[features].values, train_df["incidence"].values
X_test, y_test   = test_df[features].values, test_df["incidence"].values

# ------------------------------------------------------------
# 2. MODEL TRAINING (GLOBAL NON-LINEAR GAM)
# ------------------------------------------------------------
print("\n--- Entrenando Generalized Additive Model (GAM) ---")

# Términos s(0) a s(4): splines para cada una de las 5 variables numéricas
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4))
gam.gridsearch(X_train, y_train, progress=False)

# ------------------------------------------------------------
# 3. EVALUATION & METRICS
# ------------------------------------------------------------
gam_preds = gam.predict(X_test)

mae  = mean_absolute_error(y_test, gam_preds)
rmse = np.sqrt(mean_squared_error(y_test, gam_preds))
r2   = r2_score(y_test, gam_preds)

print(f"\n📊 Resultados GAM -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

# ------------------------------------------------------------
# 4. EXPORT SUMMARY & PREDICTIONS
# ------------------------------------------------------------
summary_df = pd.DataFrame([{
    "Model": "GAM",
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
}])

summary_df.to_excel("gam_summary.xlsx", index=False)
summary_df.to_csv("gam_summary.csv", index=False)

pred_df = pd.DataFrame({
    "country_region": test_df["country_region"],
    "year": test_df["year"],
    "y_true": y_test,
    "y_pred": gam_preds
})
pred_df.to_csv("predictions_gam.csv", index=False)

print("\n✅ Archivos guardados correctamente:")
print("   - gam_summary.xlsx")
print("   - gam_summary.csv")
print("   - predictions_gam.csv")
