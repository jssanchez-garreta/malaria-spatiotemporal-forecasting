import os
import warnings
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Silenciar warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Carga de datos
df = pd.read_csv("dataset_with_lags.csv")

# Definición de variables
target = "incidence"
features = [
    "temperature",
    "precipitation",
    "humidity",
    "precip_lag1",
    "incidence_lag1",
]
group_col = "country_region"
output_dir = "."

# ---------------------------------------------------------
# A. ENTRENAMIENTO Y GUARDADO DE MODELOS
# ---------------------------------------------------------
print("1/5. Entrenando y evaluando Mixed Effects Model...")
formula = f"{target} ~ temperature + precipitation + humidity + precip_lag1 + incidence_lag1"
me_model = smf.mixedlm(formula, df, groups=df[group_col], re_formula="~1").fit()
joblib.dump(me_model, f"{output_dir}/mixed_effects_model.joblib")

X_me = df[features]
background_me = shap.sample(X_me, 100, random_state=42)

def me_predict(X_array):
    X_df = pd.DataFrame(X_array, columns=features)
    return me_model.predict(X_df)

explainer_me = shap.Explainer(me_predict, background_me)
shap_values_me = explainer_me(X_me)

print("2/5. Entrenando y evaluando LightGBM Model...")
X_lgb = df[features].copy()
y_lgb = df[target]

lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
lgb_model.fit(X_lgb, y_lgb)
joblib.dump(lgb_model, f"{output_dir}/lightgbm_model.joblib")

explainer_lgb = shap.TreeExplainer(lgb_model)
shap_values_lgb = explainer_lgb(X_lgb)

print("3/5. Entrenando y evaluando LSTM Model...")
lookback = 3

def create_sequences(data_df, feats, target_col, lookback_steps=3):
    X_seq, y_seq = [], []
    for _, group in data_df.groupby(group_col):
        group_sorted = group.sort_values("year")
        X_vals = group_sorted[feats].values
        y_vals = group_sorted[target_col].values
        for i in range(len(group_sorted) - lookback_steps):
            X_seq.append(X_vals[i : (i + lookback_steps)])
            y_seq.append(y_vals[i + lookback_steps])
    return np.array(X_seq), np.array(y_seq)

X_lstm, y_lstm = create_sequences(df, features, target, lookback_steps=lookback)

lstm_model = Sequential([
    LSTM(32, input_shape=(lookback, len(features)), return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_lstm, y_lstm, epochs=30, batch_size=16, verbose=0)
lstm_model.save(f"{output_dir}/lstm_model.keras")

# Fijar 100 muestras exactas para evitar warnings de SHAP
X_test_lstm = X_lstm[:100]
background_2d = X_test_lstm[:, -1, :]

def lstm_predict_2d(X_2d):
    n_samples = X_2d.shape[0]
    X_seq = np.zeros((n_samples, lookback, len(features)))
    X_seq[:, -1, :] = X_2d
    return lstm_model.predict(X_seq, verbose=0).flatten()

explainer_lstm = shap.Explainer(lstm_predict_2d, background_2d)
shap_values_lstm = explainer_lstm(background_2d)

# ---------------------------------------------------------
# B. SUMMARY PLOTS (BEESWARM) - 3 Archivos PNG
# ---------------------------------------------------------
print("4/5. Generando Summary Plots (Beeswarm)...")

def save_shap_summary(shap_vals, features_data, filename):
    vals = shap_vals.values if hasattr(shap_vals, "values") else np.array(shap_vals)
    shap.summary_plot(vals, features_data, show=False, plot_type="dot")
    fig = plt.gcf()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

save_shap_summary(shap_values_me, X_me, "shap_summary_mixed_effects")
save_shap_summary(shap_values_lgb, X_lgb, "shap_summary_lightgbm")
df_lstm_features = pd.DataFrame(background_2d, columns=features)
save_shap_summary(shap_values_lstm, df_lstm_features, "shap_summary_lstm")

# ---------------------------------------------------------
# C. FEATURE IMPORTANCE & DEPENDENCE PLOTS - 9 Archivos PNG
# ---------------------------------------------------------
print("5/5. Generando Gráfico Comparativo y Plots de Dependencia Completos...")

# 1. Gráfico de Barras Comparativo (|SHAP| medio) - 1 PNG
mean_shap_me = np.abs(shap_values_me.values).mean(axis=0)
mean_shap_lgb = np.abs(shap_values_lgb.values).mean(axis=0)
mean_shap_lstm = np.abs(shap_values_lstm.values).mean(axis=0)

df_importance = pd.DataFrame({
    "Feature": features,
    "Mixed Effects": mean_shap_me,
    "LightGBM": mean_shap_lgb,
    "LSTM": mean_shap_lstm
}).set_index("Feature")

df_importance["mean_all"] = df_importance.mean(axis=1)
df_importance = df_importance.sort_values(by="mean_all", ascending=True).drop(columns=["mean_all"])

fig, ax = plt.subplots(figsize=(9, 5))

# Asignación de colores de mayor contraste:
# Mixed Effects: Azul (#1f77b4), LightGBM: Rojo (#d62728), LSTM: Verde (#2ca02c)
df_importance.plot(kind="barh", ax=ax, color=["#1f77b4", "#d62728", "#2ca02c"])

ax.set_xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("shap_feature_importance_comparison.png", dpi=300)
plt.close(fig)

# 2. Dependence Plots para TODAS las variables climáticas - 8 PNGs
climate_features = ["temperature", "precipitation", "humidity", "precip_lag1"]
models_to_plot = [
    ("LightGBM", shap_values_lgb, X_lgb),
    ("Mixed_Effects", shap_values_me, X_me),
]

for feature in climate_features:
    for model_name, shap_obj, features_ds in models_to_plot:
        vals = shap_obj.values if hasattr(shap_obj, "values") else shap_obj
        shap.dependence_plot(feature, vals, features_ds, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{feature}_{model_name}.png", dpi=300)
        plt.close(fig)

# ---------------------------------------------------------
# D. PARTIAL DEPENDENCE PLOTS (PDP) MANUALES - 5 Archivos PNG
# ---------------------------------------------------------
print("Generando Partial Dependence Plots (PDP) Comparativos...")

def compute_pdp_curve(predict_func, df_data, feature_name, grid_resolution=50):
    val_min, val_max = df_data[feature_name].min(), df_data[feature_name].max()
    grid_vals = np.linspace(val_min, val_max, grid_resolution)
    pdp_vals = []
    
    for val in grid_vals:
        df_temp = df_data.copy()
        df_temp[feature_name] = val
        preds = predict_func(df_temp.values)
        pdp_vals.append(np.mean(preds))
        
    return grid_vals, np.array(pdp_vals)

# Funciones de predicción para el PDP manual
pdp_models = [
    ("Mixed-Effects", me_predict, "#1f77b4"),  # Azul
    ("LightGBM", lambda x: lgb_model.predict(pd.DataFrame(x, columns=features)), "#d62728"),  # Rojo
    ("LSTM", lstm_predict_2d, "#2ca02c")  # Verde
]

# 1. PDP Comparativos por variable climática - 4 PNGs
for feature in climate_features:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for m_label, pred_fn, color in pdp_models:
        grid_vals, pdp_vals = compute_pdp_curve(pred_fn, X_me, feature)
        ax.plot(grid_vals, pdp_vals, label=m_label, color=color, linewidth=2)
    
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel("Partial Dependence")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"pdp_comparison_{feature}.png", dpi=300)
    plt.close(fig)

# 2. PDP específico de Temperatura para LSTM - 1 PNG
print("Generando Partial Dependence Plot para LSTM (Temperature)...")
temp_idx = features.index("temperature")
temp_range = np.linspace(X_me["temperature"].min(), X_me["temperature"].max(), 50)
base_seq = np.mean(X_lstm, axis=0)

pdp_preds = []
for t in temp_range:
    seq_temp = base_seq.copy()
    seq_temp[:, temp_idx] = t
    pred = lstm_model.predict(seq_temp[np.newaxis, :, :], verbose=0)[0, 0]
    pdp_preds.append(pred)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(temp_range, pdp_preds, color="#7570b3", linewidth=2.5)
ax.set_xlabel("Temperature")
ax.set_ylabel("Predicted Incidence")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("lstm_pdp_temperature.png", dpi=300)
plt.close(fig)

print("\n¡Proceso XAI completo ejecutado exitosamente! Se han generado exactamente 17 archivos .png.")
