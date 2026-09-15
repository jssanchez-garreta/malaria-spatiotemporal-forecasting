# Uncertainty-Aware and Interpretable Machine Learning for Multi-Regional Malaria Incidence Forecasting: A 25-Year Eco-Climatic Benchmark in Sub-Saharan Africa

<a href="https://doi.org/10.5281/zenodo.12774578"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.12774578.svg" alt="DOI"></a>

Official repository for the multi-regional benchmark study evaluating statistical, tree-based, and deep neural network models coupled with Split Conformal Prediction and Explainable AI (XAI) for malaria forecasting across 340 administrative regions in sub-Saharan Africa (2000–2024).

---

## Key Highlights

- **Multi-Regional Benchmark:** Comprehensive evaluation of 9 predictive architectures across 340 administrative regions spanning 25 years.
- **Guaranteed Uncertainty Quantification:** Integration of Split Conformal Prediction yielding distribution-free 95% prediction intervals with >98.9% empirical coverage.
- **Mechanistic Interpretability:** SHAP and PDP analyses isolating historical persistence and non-linear climate forcing (24°C–27°C thermal optimal zones).
- **Eco-Climatic Stratification:** Model evaluation across diverse climatic regimes (Sahelian vs. Highland zones).

---

## Repository Structure

- **data/**: Preprocessed epidemiological and eco-climatic datasets
- **models/**: Implementation of predictive paradigms (statistical, tree-based, deep learning)
- **conformal/**: Split Conformal Prediction algorithms & interval scoring
- **xai/**: SHAP value extraction & PDP visualization scripts
- **requirements.txt**: Environment dependencies

---

## Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/jssanchez-garreta/malaria-spatiotemporal-forecasting.git
cd malaria-spatiotemporal-forecasting
pip install -r requirements.txt
```

---

## Dataset Overview

The study leverages monthly epidemiological and eco-climatic observations (2000–2024):
- **Target Variable:** Malaria incidence rate per 1,000 population.
- **Predictors:** Climate variables (temperature, precipitation, LST, humidity, NDVI), temporal lags (lag1–lag12), and regional encodings.

---

## Model Benchmarking Summary

| Paradigm | Model | RMSE | MAE | R2 |
| :--- | :--- | :--- | :--- | :--- |
| Deep Learning | LSTM | 19.62 | 14.12 | 0.9689 |
| Deep Learning | GRU | 20.15 | 14.50 | 0.9654 |
| Deep Learning | Temporal Transformer | 20.88 | 15.02 | 0.9610 |
| Tree Ensembles | LightGBM | 20.04 | 14.28 | 0.9671 |
| Tree Ensembles | XGBoost | 20.45 | 14.65 | 0.9632 |
| Tree Ensembles | Random Forest | 21.30 | 15.10 | 0.9580 |
| Statistical Panel | Mixed-Effects | 20.90 | 13.66 | 0.9612 |
| Statistical Panel | GAM | 21.05 | 13.95 | 0.9601 |
| Baseline | ARIMA | 28.40 | 19.80 | 0.8920 |

---

## Citation & License

If you use this dataset or codebase, please cite our study:

```bibtex
@article{sanchez2026malaria,
  title={Uncertainty-aware and interpretable machine learning for multi-regional malaria incidence forecasting: A 25-year eco-climatic benchmark in sub-Saharan Africa},
  author={Sánchez-Marqués, R., Sánchez, J.S.},
  journal={Computers in Biology and Medicine},
  year={2026}
}
```

Distributed under the Apache 2.0 License. See LICENSE for more information.
