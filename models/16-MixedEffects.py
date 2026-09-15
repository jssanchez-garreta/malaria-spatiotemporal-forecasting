# ============================================================
# MIXED-EFFECTS PANEL MODEL - INDEPENDENT PIPELINE
# ============================================================

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# ------------------------------------------------------------
# 2. MODEL TRAINING (GLOBAL MIXED-EFFECTS MODEL)
# ------------------------------------------------------------
print("\n--- Entrenando Modelo de Efectos Mixtos (Global) ---")

# Fórmula: Incidencia explicada por clima y lags, con intercepto aleatorio por región
formula = "incidence ~ temperature + precipitation + humidity + precip_lag1 + incidence_lag1"

model = smf.mixedlm(
    formula=formula,
    data=train_df,
    groups=train_df["country_region"]
)

mixed_fit = model.fit()

# ------------------------------------------------------------
# 3. EVALUATION & METRICS
# ------------------------------------------------------------
# Predicción global sobre test
test_preds = mixed_fit.predict(test_df)

mae  = mean_absolute_error(test_df["incidence"], test_preds)
rmse = np.sqrt(mean_squared_error(test_df["incidence"], test_preds))
r2   = r2_score(test_df["incidence"], test_preds)

print(f"\n📊 Resultados Mixed-Effects -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

# ------------------------------------------------------------
# 4. EXPORT SUMMARY & PREDICTIONS
# ------------------------------------------------------------
summary_df = pd.DataFrame([{
    "Model": "Mixed-Effects Panel",
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
}])

summary_df.to_excel("mixed_effects_summary.xlsx", index=False)
summary_df.to_csv("mixed_effects_summary.csv", index=False)

pred_df = pd.DataFrame({
    "country_region": test_df["country_region"],
    "year": test_df["year"],
    "y_true": test_df["incidence"].values,
    "y_pred": test_preds.values
})
pred_df.to_csv("predictions_mixed_effects.csv", index=False)

print("\n✅ Archivos guardados correctamente:")
print("   - mixed_effects_summary.xlsx")
print("   - mixed_effects_summary.csv")
print("   - predictions_mixed_effects.csv")