# Uncertainty-Aware and Interpretable Machine Learning for Multi-Regional Malaria Incidence Forecasting: A 25-Year Eco-Climatic Benchmark in Sub-Saharan Africa

Official repository for the multi-regional benchmark study evaluating statistical, tree-based, and deep neural network models coupled with Split Conformal Prediction and Explainable AI (XAI) for malaria forecasting across 340 administrative regions in 20 sub-Saharan African countries (2000–2024).

---

## 📌 Key Highlights

* **Multi-Regional Benchmark:** Comprehensive evaluation of 9 predictive architectures (ARIMA, GAM, Mixed-Effects, Random Forest, XGBoost, LightGBM, LSTM, GRU, Temporal Transformer) across 340 administrative regions spanning 25 years.
* **Guaranteed Uncertainty Quantification:** Integration of Split Conformal Prediction yielding distribution-free 95% prediction intervals with >98.9% empirical coverage and narrow interval width (MPIW ≤ 132.66).
* **Mechanistic Interpretability:** SHAP and Partial Dependence Plot (PDP) analyses isolating historical persistence (`incidence_lag1`) and non-linear climate forcing, including optimal transmission thermal zones (24°C–27°C).
* **Eco-Climatic Stratification:** Model evaluation across diverse climatic regimes, highlighting performance trade-offs in seasonal Sahelian versus topographically complex Highland regions.

---

## 🛠️ Repository Structure

```text
├── data/                  # Preprocessed epidemiological and eco-climatic datasets
├── models/                # Implementation of predictive paradigms
│   ├── statistical/       # ARIMA, GAM, Linear Mixed-Effects
│   ├── tree_based/        # Random Forest, XGBoost, LightGBM
│   └── deep_learning/     # LSTM, GRU, Temporal Transformer
├── conformal/             # Split Conformal Prediction algorithms & interval scoring
├── xai/                   # SHAP value extraction & PDP visualization scripts
├── requirements.txt       # Environment dependencies and version specs
└── README.md              # Project documentation

---

```text
## 🚀 Getting Started
Prerequisites
Ensure you have Python 3.9+ installed. Clone the repository and install the required dependencies:

git clone [https://github.com/jssanchez-garreta/malaria-spatiotemporal-forecasting.git](https://github.com/jssanchez-garreta/malaria-spatiotemporal-forecasting.git)
cd malaria-spatiotemporal-forecasting
pip install -r requirements.txt

---

## 📊 Dataset Overview
The study leverages monthly epidemiological and eco-climatic observations spanning 2000–2024:

Target Variable: Malaria incidence rate per 1,000 population.

Predictor Domains:

Climatic & Environmental: Temperature (mean, min, max), precipitation, land surface temperature (LST), relative humidity, and vegetation indices (NDVI/EVI).

Temporal Lags: Autoregressive features (lag1, ..., lag12) capturing epidemiological memory.

Spatial Identifiers: Eco-climatic region encodings and spatial adjacency indicators across 340 administrative units.

---

## 📜 Citation & License
If you use this dataset, codebase, or benchmarking framework in your research, please cite our study:

@article{sanchez2026malaria,
  title={Uncertainty-aware and interpretable machine learning for multi-regional malaria incidence forecasting: A 25-year eco-climatic benchmark in sub-Saharan Africa},
  author={S{\'a}nchez-Garreta, Josep and others},
  journal={Computers in Biology and Medicine},
  year={2026}
}
