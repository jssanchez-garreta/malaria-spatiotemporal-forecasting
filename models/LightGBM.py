# ============================================================
# LIGHTGBM REGRESSOR - INDEPENDENT PIPELINE
# ============================================================

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
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
print("\n--- Entrenando LightGBM Regressor ---")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
)

# ------------------------------------------------------------
# 3. EVALUATION & METRICS
# ------------------------------------------------------------
lgb_preds = lgb_model.predict(X_test)

mae  = mean_absolute_error(y_test, lgb_preds)
rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
r2   = r2_score(y_test, lgb_preds)

print(f"\n📊 Resultados LightGBM -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

# ------------------------------------------------------------
# 4. SAVE SUMMARY METRICS (.EXCEL & .CSV)
# ------------------------------------------------------------
summary_df = pd.DataFrame([{
    "Model": "LightGBM",
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4)
}])

summary_df.to_excel("lgb_summary.xlsx", index=False)
summary_df.to_csv("lgb_summary.csv", index=False)

# ------------------------------------------------------------
# 5. SAVE POINTWISE PREDICTIONS (FOR TESTS & PLOTS)
# ------------------------------------------------------------
pred_df = pd.DataFrame({
    "country_region": test_df["country_region"],
    "year": test_df["year"],
    "y_true": y_test.values,
    "y_pred": lgb_preds
})
pred_df.to_csv("predictions_lgb.csv", index=False)

print("\n✅ Archivos guardados correctamente:")
print("   - lgb_summary.xlsx")
print("   - lgb_summary.csv")
print("   - predictions_lgb.csv")
