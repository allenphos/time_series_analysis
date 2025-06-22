# 📊 Store Item Demand Forecasting

## 🌟 Goal
Predict daily sales for 50 items across 10 stores for the next 30 and 90 days using various time series and machine learning models.

---

## 📂 Dataset
- **Source**: [Kaggle - Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting/data)
- **Size**: 913,000 rows over a 5-year period (2013-01-01 to 2017-12-31)
- **Fields**: `date`, `store`, `item`, `sales`

---

## 🚀 Workflow Overview

### 1. **Exploratory Data Analysis (EDA)**
- Time series visualization with Plotly
- Boxplots to assess inter-store variation
- Seasonal trend discovery (summer and Christmas peaks)

### 2. **Preprocessing**
- Convert `date` to datetime, set as index
- Seasonal decomposition with `statsmodels`
- Augmented Dickey-Fuller (ADF) test for stationarity (p < 0.05)

### 3. **Modeling Approaches**

#### Baseline Models
- `NaiveSeasonal(K=14)`
- `NaiveDrift`
- Combined: Drift + Seasonality (MAPE ~29%)

#### Classical Time Series Models
- **ARIMA** (MAPE ~39.5%)
- **AutoARIMA** (MAPE ~38%)
- **SARIMAX with exogenous vars** (MAPE ~38.8%)
- **Exponential Smoothing** (MAPE ~39%)

#### Machine Learning
- **XGBoost (XGBModel)** from Darts
  - Feature engineering: cyclic temporal features
  - Hyperparameter tuning improved MAPE to ~25.9%

#### Deep Learning
- **RNN (LSTM)** with Darts
  - Default LSTM: MAPE ~36.9%
  - Scaled + covariates: improved to ~29.9%

#### Prophet
- Daily seasonality enabled
- Best performance: **MAPE ~23.8%** on validation, **20.4%** on backtest

### 4. **Backtesting**
- 1-year backtest with 30-day rolling forecasts
- Residual analysis: normally distributed around 0

### 5. **Forecasting Output**
- Forecasts generated for all 50 items across 10 stores
- Two horizons: 30 days and 90 days

---

## 📊 Visualization
- Interactive dashboards (Plotly)
  - Forecasted sales per item/store
  - Actual vs forecasted plots by item and store
  - Interactive filters with `ipywidgets`
- Residual histograms

---

## 📊 Evaluation Metrics
- **Mean Absolute Percentage Error (MAPE)** across models:
  - Prophet: **23.8%** (validation), **20.4%** (backtest)
  - XGBoost: 25.9%
  - RNN (scaled): 29.9%
  - AutoARIMA, ARIMA, SARIMA: ~38-39%

---

## 🔧 Tech Stack
- Python
- Libraries: `pandas`, `darts`, `prophet`, `plotly`, `statsmodels`, `xgboost`, `ipywidgets`
- Notebooks: Google Colab

---

## 📊 Key Learnings
- Prophet consistently outperformed other models for multivariate time series
- Feature engineering and scaling improve ML model performance
- Backtesting and residual analysis are crucial for robust evaluation

---

## Run on Google Colab

You can run the main notebook interactively in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/username/repo/blob/main/notebooks/timeseries.ipynb](https://colab.research.google.com/drive/1C8r2Jutet8WzaGQjlSPglQA-zPPjj89s?usp=drive_open#scrollTo=eTmgpP3vZK5T))
---

## 🚧 Future Work
- Hyperparameter optimization (Optuna)
- Cross-store learning with transfer learning
- Add external regressors (holidays, promotions)
- Deploy as API or dashboard for retail planners

---

## 🛌 Author
- Project by: *Anastasiia Alyoshkina*
- Contact: *anastasia.alshkn@gmail.com*
