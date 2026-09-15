# ============================================================
# ARIMA / AUTO-ARIMA PIPELINE FOR PANEL DATA (COMPLETE EXPORT)
# ============================================================

import os
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

# ------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------
file_name = "final_dataset.csv"

if os.path.exists("train_df.csv") and os.path.exists("test_df.csv"):
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
# 2. ENTRENAMIENTO E INFERENCIA POR REGIÓN
# ------------------------------------------------------------
print("\n--- Entrenando ARIMA por región ---")

predictions_list = []
unique_regions = test_df["country_region"].unique()

for region in unique_regions:
    region_train = train_df[train_df["country_region"] == region].sort_values("year")
    region_test  = test_df[test_df["country_region"] == region].sort_values("year")
    
    n_test = len(region_test)
    if n_test == 0:
        continue
        
    y_train = region_train["incidence"].values
    y_test  = region_test["incidence"].values
    
    preds = []
    try:
        if len(y_train) >= 4:
            model = ARIMA(y_train, order=(1, 1, 0))
            fitted = model.fit()
            preds = fitted.forecast(steps=n_test)
        else:
            raise ValueError("Insuficiente historial de entrenamiento")
    except Exception:
        # Fallback 1: Persistencia / Último valor observado
        last_val = y_train[-1] if len(y_train) > 0 else np.mean(train_df["incidence"])
        preds = np.full(n_test, fill_value=last_val)

    # Reemplazar valores nulos o infinitos si aparecieran
    preds = np.nan_to_num(preds, nan=np.mean(y_train) if len(y_train) > 0 else 0.0)

    for idx, (_, row) in enumerate(region_test.iterrows()):
        predictions_list.append({
            "country_region": row["country_region"],
            "year": row["year"],
            "y_true": row["incidence"],
            "y_pred": preds[idx]
        })

# ------------------------------------------------------------
# 3. EVALUACIÓN Y EXPORTACIÓN DE RESULTADOS
# ------------------------------------------------------------
pred_df = pd.DataFrame(predictions_list)

y_true = pred_df["y_true"].values
y_pred = pred_df["y_pred"].values

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print(f"\n📊 Resultados ARIMA -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

# Guardar resumen de métricas (.csv y .xlsx)
summary_df = pd.DataFrame([{
    "Model": "ARIMA",
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
}])

summary_df.to_csv("arima_summary.csv", index=False)
summary_df.to_excel("arima_summary.xlsx", index=False)

# Guardar predicciones completas (.csv)
pred_df.to_csv("predictions_arima.csv", index=False)

print("\n✅ Archivos guardados correctamente:")
print("   - predictions_arima.csv")
print("   - arima_summary.csv")
print("   - arima_summary.xlsx")
