# ============================================================
# XGBOOST REGRESSOR - INDEPENDENT PIPELINE
# ============================================================

import os
import pandas as pd
import numpy as np
import xgboost as xgb
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
    val_df   = pd.read_csv("val_df.csv")
    test_df  = pd.read_csv("test_df.csv")
else:
    df = pd.read_csv(file_name)
    df = df.sort_values(by=["country_region", "year"]).reset_index(drop=True)
    df["precip_lag1"] = df.groupby("country_region")["precipitation"].shift(1)
    df["incidence_lag1"] = df.groupby("country_region")["incidence"].shift(1)
    df = df.dropna().reset_index(drop=True)

    train_df = df[df["year"] <= 2016].copy()
    val_df   = df[(df["year"] >= 2017) & (df["year"] <= 2020)].copy()
    test_df  = df[df["year"] >= 2021].copy()

features = ["temperature", "precipitation", "humidity", "precip_lag1", "incidence_lag1"]

X_train, y_train = train_df[features], train_df["incidence"]
X_val, y_val     = val_df[features], val_df["incidence"]
X_test, y_test   = test_df[features], test_df["incidence"]

# ------------------------------------------------------------
# 2. MODEL TRAINING
# ------------------------------------------------------------
print("\n--- Entrenando XGBoost Regressor ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=15
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ------------------------------------------------------------
# 3. EVALUATION & METRICS
# ------------------------------------------------------------
xgb_preds = xgb_model.predict(X_test)

mae  = mean_absolute_error(y_test, xgb_preds)
rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
r2   = r2_score(y_test, xgb_preds)

print(f"\n📊 Resultados XGBoost -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

# ------------------------------------------------------------
# 4. SAVE SUMMARY METRICS (.EXCEL & .CSV)
# ------------------------------------------------------------
summary_df = pd.DataFrame([{
    "Model": "XGBoost",
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
}])

summary_df.to_excel("xgb_summary.xlsx", index=False)
summary_df.to_csv("xgb_summary.csv", index=False)

# ------------------------------------------------------------
# 5. SAVE POINTWISE PREDICTIONS (FOR TESTS & PLOTS)
# ------------------------------------------------------------
pred_df = pd.DataFrame({
    "country_region": test_df["country_region"],
    "year": test_df["year"],
    "y_true": y_test.values,
    "y_pred": xgb_preds
})
pred_df.to_csv("predictions_xgb.csv", index=False)

print("\n✅ Archivos guardados correctamente:")
print("   - xgb_summary.xlsx")
print("   - xgb_summary.csv")
print("   - predictions_xgb.csv")
