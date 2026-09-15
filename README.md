# Uncertainty-Aware and Interpretable Machine Learning for Multi-Regional Malaria Incidence Forecasting: A 25-Year Eco-Climatic Benchmark in Sub-Saharan Africa

Official repository for the multi-regional benchmark study evaluating statistical, tree-based, and deep neural network models coupled with Split Conformal Prediction and Explainable AI (XAI) for malaria forecasting across 340 administrative regions in 20 sub-Saharan African countries (2000–2024).

---

## 📌 Key Highlights

* **Multi-Regional Benchmark:** Comprehensive evaluation of 9 predictive architectures (ARIMA, GAM, Mixed-Effects, Random Forest, XGBoost, LightGBM, LSTM, GRU, Temporal Transformer) across 340 administrative regions spanning 25 years.
* **Guaranteed Uncertainty Quantification:** Integration of Split Conformal Prediction yielding distribution-free 95% prediction intervals with >98.9% empirical coverage and narrow interval width ($\text{MPIW} \le 132.66$).
* **Mechanistic Interpretability:** SHAP and Partial Dependence Plot (PDP) analyses isolating historical persistence ($\text{incidence\_lag1}$) and non-linear climate forcing, including optimal transmission thermal zones ($24^\circ\text{C}$–$27^\circ\text{C}$).
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

## Contact

For questions or further information:  
J. Salvador Sánchez (sanchez AT uji.es)

## License

This repository is provided for academic and research purposes.
