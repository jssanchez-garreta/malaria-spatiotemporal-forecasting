!pip install pygam

import os
import warnings
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb
from pygam import LinearGAM, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras import layers, models

warnings.filterwarnings("ignore")

# 1. Carga de datos
df = pd.read_csv("dataset_with_lags.csv")

target = "incidence"
features = [
    "temperature",
    "precipitation",
    "humidity",
    "precip_lag1",
    "incidence_lag1",
]
group_col = "country_region"
alpha = 0.05  # Nivel de confianza del 95%

# 2. Partición temporal (Train 60%, Calibration 20%, Test 20%)
df = df.sort_values(by=["year"]).reset_index(drop=True)
n_total = len(df)
n_train = int(n_total * 0.60)
n_calib = int(n_total * 0.20)

df_train = df.iloc[:n_train].copy()
df_calib = df.iloc[n_train : n_train + n_calib].copy()
df_test = df.iloc[n_train + n_calib :].copy()

X_train, y_train = df_train[features], df_train[target]
X_calib, y_calib = df_calib[features], df_calib[target]
X_test, y_test = df_test[features], df_test[target]


# ---------------------------------------------------------
# PREPARACIÓN PARA MODELOS DE SECUENCIA (LSTM, GRU, TRANSFORMER)
# ---------------------------------------------------------
def create_sequences(data_df, lookback_steps=3):
    X_seq, y_seq = [], []
    for _, group in data_df.groupby(group_col):
        group_sorted = group.sort_values("year")
        X_vals = group_sorted[features].values
        y_vals = group_sorted[target].values
        for i in range(len(group_sorted) - lookback_steps):
            X_seq.append(X_vals[i : (i + lookback_steps)])
            y_seq.append(y_vals[i + lookback_steps])
    return np.array(X_seq), np.array(y_seq)


lookback = 3
X_tr_seq, y_tr_seq = create_sequences(df_train, lookback)
X_cal_seq, y_cal_seq = create_sequences(df_calib, lookback)
X_te_seq, y_te_seq = create_sequences(df_test, lookback)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_tr_flat = X_tr_seq.reshape(-1, len(features))
scaler_X.fit(X_tr_flat)
scaler_y.fit(y_tr_seq.reshape(-1, 1))


def scale_seq(X_seq):
    samples, steps, feats = X_seq.shape
    X_scaled = scaler_X.transform(X_seq.reshape(-1, feats))
    return X_scaled.reshape(samples, steps, feats)


X_tr_seq_sc = scale_seq(X_tr_seq)
X_cal_seq_sc = scale_seq(X_cal_seq)
X_te_seq_sc = scale_seq(X_te_seq)
y_tr_seq_sc = scaler_y.transform(y_tr_seq.reshape(-1, 1)).flatten()

# ---------------------------------------------------------
# A. ENTRENAMIENTO DE LOS 9 MODELOS OFICIALES
# ---------------------------------------------------------
print("Entrenando los 9 modelos oficiales...")

# 1. ARIMA (Por región/grupo)
arima_preds_calib, arima_preds_test = [], []


# Función helper para ajustar ARIMA por región
def fit_predict_arima(df_tr, df_eval):
    preds = []
    for region, grp_eval in df_eval.groupby(group_col):
        grp_tr = df_tr[df_tr[group_col] == region]
        y_tr_reg = grp_tr[target].values
        try:
            model = ARIMA(y_tr_reg, order=(1, 1, 0)).fit()
            p = model.forecast(steps=len(grp_eval))
        except:
            p = np.full(len(grp_eval), y_tr_reg.mean() if len(y_tr_reg) > 0 else 0)
        preds.extend(p)
    return np.array(preds)


arima_preds_calib = fit_predict_arima(df_train, df_calib)
arima_preds_test = fit_predict_arima(
    pd.concat([df_train, df_calib]), df_test
)

# 2. Mixed Effects Model
formula = f"{target} ~ " + " + ".join(features)
me = smf.mixedlm(
    formula, df_train, groups=df_train[group_col], re_formula="~1"
).fit()

# 3. GAM (Generalized Additive Model)
gam_terms = s(0) + s(1) + s(2) + s(3) + s(4)
gam = LinearGAM(gam_terms).fit(X_train.values, y_train.values)

# 4. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(
    X_train, y_train
)

# 5. XGBoost
xgb_mod = xgb.XGBRegressor(
    n_estimators=100, learning_rate=0.05, random_state=42
).fit(X_train, y_train)

# 6. LightGBM
lgb_mod = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1
).fit(X_train, y_train)

# 7. GRU
gru = models.Sequential(
    [
        layers.GRU(
            32, input_shape=(lookback, len(features)), return_sequences=False
        ),
        layers.Dropout(0.2),
        layers.Dense(1),
    ]
)
gru.compile(optimizer="adam", loss="mse")
gru.fit(X_tr_seq_sc, y_tr_seq_sc, epochs=60, batch_size=16, verbose=0)

# 8. LSTM
lstm = models.Sequential(
    [
        layers.LSTM(
            32, input_shape=(lookback, len(features)), return_sequences=False
        ),
        layers.Dropout(0.2),
        layers.Dense(1),
    ]
)
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_tr_seq_sc, y_tr_seq_sc, epochs=60, batch_size=16, verbose=0)


# 9. Temporal Transformer
def build_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(32)(inputs)
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs, outputs)


transformer = build_transformer((lookback, len(features)))
transformer.compile(optimizer="adam", loss="mse")
transformer.fit(X_tr_seq_sc, y_tr_seq_sc, epochs=60, batch_size=16, verbose=0)


# Dispatcher de predicciones
def get_predictions(name, X_df, X_seq_sc, eval_type="test"):
    if name == "ARIMA":
        return arima_preds_calib if eval_type == "calib" else arima_preds_test
    elif name == "Mixed-Effects":
        return me.predict(X_df).values
    elif name == "GAM":
        return gam.predict(X_df.values)
    elif name == "Random Forest":
        return rf.predict(X_df)
    elif name == "XGBoost":
        return xgb_mod.predict(X_df)
    elif name == "LightGBM":
        return lgb_mod.predict(X_df)
    elif name == "GRU":
        return scaler_y.inverse_transform(
            gru.predict(X_seq_sc, verbose=0)
        ).flatten()
    elif name == "LSTM":
        return scaler_y.inverse_transform(
            lstm.predict(X_seq_sc, verbose=0)
        ).flatten()
    elif name == "Temporal Transformer":
        return scaler_y.inverse_transform(
            transformer.predict(X_seq_sc, verbose=0)
        ).flatten()


# ---------------------------------------------------------
# B. CONFORMAL PREDICTION (SPLIT CONFORMAL)
# ---------------------------------------------------------
model_names = [
    "ARIMA",
    "Mixed-Effects",
    "GAM",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "GRU",
    "LSTM",
    "Temporal Transformer",
]

results = []
test_predictions_dict = {}

for name in model_names:
    is_seq = name in ["GRU", "LSTM", "Temporal Transformer"]
    y_calib_actual = y_cal_seq if is_seq else y_calib.values
    y_test_actual = y_te_seq if is_seq else y_test.values

    # Predicciones calibración y cuantiles
    pred_calib = get_predictions(name, X_calib, X_cal_seq_sc, eval_type="calib")
    calib_scores = np.abs(y_calib_actual - pred_calib)

    n_cal = len(calib_scores)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(1.0, q_level)
    q_hat = np.quantile(calib_scores, q_level)

    # Predicciones test e intervalos
    pred_test = get_predictions(name, X_test, X_te_seq_sc, eval_type="test")
    lower_bound = pred_test - q_hat
    upper_bound = pred_test + q_hat

    covered = (y_test_actual >= lower_bound) & (y_test_actual <= upper_bound)
    empirical_coverage = np.mean(covered) * 100
    mpiw = np.mean(upper_bound - lower_bound)

    # Winkler Score
    delta = upper_bound - lower_bound
    winkler = delta + (2 / alpha) * (lower_bound - y_test_actual) * (
        y_test_actual < lower_bound
    ) + (2 / alpha) * (y_test_actual - upper_bound) * (
        y_test_actual > upper_bound
    )
    mean_winkler = np.mean(winkler)

    results.append(
        {
            "Model": name,
            "q_hat": round(q_hat, 4),
            "Empirical Coverage (%)": round(empirical_coverage, 2),
            "MPIW": round(mpiw, 4),
            "Winkler Score": round(mean_winkler, 4),
        }
    )

    test_predictions_dict[name] = {
        "y_true": y_test_actual,
        "pred": pred_test,
        "lower": lower_bound,
        "upper": upper_bound,
    }

# ---------------------------------------------------------
# C. GUARDAR Y GRAFICAR PANEL 3x3
# ---------------------------------------------------------
df_results = pd.DataFrame(results).sort_values(by="MPIW", ascending=True)
print("\n--- RESUMEN CONFORMAL PREDICTION (9 MODELOS REALES) ---")
print(df_results.to_string(index=False))

df_results.to_csv("conformal_prediction_results_9_models_real.csv", index=False)

# Exportar tabla formateada a LaTeX
df_results.to_latex(
    "conformal_prediction_results_9_models_real.tex",
    index=False,
    float_format="%.4f",
    caption="Conformal Prediction results (95% Confidence Interval) for the 9 evaluated models.",
    label="tab:conformal_prediction_9_models",
)

plt.figure(figsize=(15, 11))
subset_len = 100

for i, name in enumerate(model_names, 1):
    plt.subplot(3, 3, i)
    data = test_predictions_dict[name]
    x_axis = np.arange(subset_len)

    plt.plot(
        x_axis,
        data["y_true"][:subset_len],
        "k.",
        label="Real",
        alpha=0.6,
        markersize=4,
    )
    plt.plot(
        x_axis,
        data["pred"][:subset_len],
        color="#2b5c8f",
        linewidth=1,
        label="Pred",
    )
    plt.fill_between(
        x_axis,
        data["lower"][:subset_len],
        data["upper"][:subset_len],
        color="#2b5c8f",
        alpha=0.2,
        label="95% CI",
    )

    plt.title(f"{name}", fontsize=10, fontweight="bold")
    plt.xlabel("Test Sample", fontsize=8)
    plt.ylabel("Incidence", fontsize=8)
    if i == 1:
        plt.legend(loc="upper left", fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("conformal_prediction_9_models_real_comparison.png", dpi=300)
plt.close()
