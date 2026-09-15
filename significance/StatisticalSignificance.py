# ============================================================
# PAIRWISE STATISTICAL SIGNIFICANCE TESTS (ALL vs ALL)
# ============================================================

import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

def diebold_mariano_test(y_true, y_pred1, y_pred2):
    """
    Test de Diebold-Mariano con función de pérdida de Error Absoluto.
    """
    e1 = np.abs(y_true - y_pred1)
    e2 = np.abs(y_true - y_pred2)
    d = e1 - e2
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    if var_d == 0 or np.isnan(var_d):
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

# ------------------------------------------------------------
# 1. CARGAR PREDICCIONES DE TODOS LOS MODELOS
# ------------------------------------------------------------
model_files = {
    "XGBoost": "predictions_xgb.csv",
    "LightGBM": "predictions_lgb.csv",
    "Random Forest": "predictions_rf.csv",
    "GRU": "predictions_gru.csv",
    "LSTM": "predictions_lstm.csv",
    "Transformer": "predictions_transformer.csv",
    "ARIMA": "predictions_arima.csv",
    "Mixed Effects": "predictions_mixed_effects.csv",
    "GAM": "predictions_gam.csv"
}

loaded_models = {}

print("\n--- CARGANDO PREDICCIONES PARA MATRIZ TODOS VS TODOS ---")
for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        df_mod = pd.read_csv(file_path)
        df_mod.columns = df_mod.columns.str.strip()
        loaded_models[model_name] = df_mod
        print(f"✅ Cargado: {model_name} ({len(df_mod)} muestras)")
    else:
        print(f"⚠️ Omite: {file_path} (No encontrado)")

model_names = list(loaded_models.keys())

if len(model_names) < 2:
    raise ValueError("❌ Se necesitan al menos 2 modelos para realizar comparativas estadísticas.")

# ------------------------------------------------------------
# 2. COMPARACIÓN TODOS CONTRA TODOS (PAIRWISE COMBINATIONS)
# ------------------------------------------------------------
pairwise_results = []

# Inicializar matrices cuadradas para p-valores
wilcoxon_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
dm_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)

# Diagonal principal (comparar modelo consigo mismo)
for name in model_names:
    wilcoxon_matrix.loc[name, name] = 1.0
    dm_matrix.loc[name, name] = 1.0

# Bucle sobre cada par único de modelos
for model_A, model_B in combinations(model_names, 2):
    df_A = loaded_models[model_A]
    df_B = loaded_models[model_B]
    
    # Alineación por región y año
    merged = pd.merge(
        df_A[["country_region", "year", "y_true", "y_pred"]],
        df_B[["country_region", "year", "y_pred"]],
        on=["country_region", "year"],
        suffixes=(f"_{model_A}", f"_{model_B}")
    ).dropna()
    
    n_samples = len(merged)
    if n_samples < 5:
        continue
        
    y_true = merged["y_true"].values
    y_pred_A = merged[f"y_pred_{model_A}"].values
    y_pred_B = merged[f"y_pred_{model_B}"].values
    
    e_A = np.abs(y_true - y_pred_A)
    e_B = np.abs(y_true - y_pred_B)
    
    # Wilcoxon
    try:
        w_stat, w_p = stats.wilcoxon(e_A, e_B)
    except Exception:
        w_stat, w_p = np.nan, 1.0
        
    # Diebold-Mariano
    dm_stat, dm_p = diebold_mariano_test(y_true, y_pred_A, y_pred_B)
    
    # Guardar en lista de pares
    pairwise_results.append({
        "Model_A": model_A,
        "Model_B": model_B,
        "N_Samples": n_samples,
        "Wilcoxon_Stat": round(w_stat, 2) if not np.isnan(w_stat) else "N/A",
        "Wilcoxon_p_value": f"{w_p:.5e}",
        "Wilcoxon_Sig": "Sí (p < 0.05)" if w_p < 0.05 else "No",
        "DM_Stat": round(dm_stat, 4),
        "DM_p_value": f"{dm_p:.5e}",
        "DM_Sig": "Sí (p < 0.05)" if dm_p < 0.05 else "No"
    })
    
    # Rellenar matrices simétricas
    wilcoxon_matrix.loc[model_A, model_B] = w_p
    wilcoxon_matrix.loc[model_B, model_A] = w_p
    
    dm_matrix.loc[model_A, model_B] = dm_p
    dm_matrix.loc[model_B, model_A] = dm_p

# ------------------------------------------------------------
# 3. EXPORTAR RESULTADOS PAREADOS Y MATRICES
# ------------------------------------------------------------
pairs_df = pd.DataFrame(pairwise_results)
pairs_df.to_excel("statistical_significance_pairs.xlsx", index=False)
pairs_df.to_csv("statistical_significance_pairs.csv", index=False)

wilcoxon_matrix.to_excel("wilcoxon_p_values_matrix.xlsx")
dm_matrix.to_excel("dm_p_values_matrix.xlsx")

print("\n✅ Archivos guardados correctamente:")
print("   - statistical_significance_pairs.xlsx (Lista exhaustiva de pares)")
print("   - statistical_significance_pairs.csv")
print("   - wilcoxon_p_values_matrix.xlsx (Matriz cuadrada N x N de p-valores Wilcoxon)")
print("   - dm_p_values_matrix.xlsx (Matriz cuadrada N x N de p-valores Diebold-Mariano)")
