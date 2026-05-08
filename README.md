# Volatility-Forecasting

A daily 1-day-ahead volatility forecasting system for SPY (S&P 500 ETF) covering 2008–2026,
comparing classical econometric baselines against modern ML approaches using a rigorous
walk-forward out-of-sample evaluation framework.

**Core research question:** Can ML models with engineered features beat classical volatility
benchmarks (HAR-RV, GARCH) on out-of-sample SPY daily volatility forecasts?

**Short answer:** No — GARCH(1,1) wins on the primary QLIKE metric. XGBoost comes closest
on log-MSE. Ridge and Lasso fail catastrophically due to target-space sensitivity.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Results](#results)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Volatility Estimators](#volatility-estimators)
6. [Feature Engineering](#feature-engineering)
7. [Models](#models)
8. [Walk-Forward Validation](#walk-forward-validation)
9. [Loss Functions](#loss-functions)
10. [Reproducing Results](#reproducing-results)
11. [Dependencies](#dependencies)
12. [References](#references)

---

## Project Overview

This project implements a full volatility forecasting pipeline for SPY daily returns from
**January 2, 2008 to April 28, 2026** (4,619 trading days). The system:

- Computes four range-based volatility estimators (Yang-Zhang, Garman-Klass, Parkinson, Rogers-Satchell)
- Engineers 22 strictly causal features with automated no-lookahead validation
- Trains 6 models across 13 annual walk-forward folds (~3,000 out-of-sample predictions per model)
- Evaluates with QLIKE, MSE on log-variance, and MAE on volatility
- Saves per-fold predictions, loss summaries, and diagnostic figures

Everything is fully reproducible: global `seed=22`, all splits respect temporal order, and
hyperparameter search is nested within each training fold.

---

## Results

### Summary Table

| Model | QLIKE (↓) | QLIKE Std | MSE-log (↓) | MSE-log Std | MAE-vol (↓) | MAE-vol Std |
|---|---|---|---|---|---|---|
| **GARCH(1,1)** | **69.64** | 249.18 | 1.338 | 0.473 | **0.00546** | 0.00660 |
| HAR-RV | 74.79 | 266.41 | 6.008 | 16.600 | 0.267 | 0.946 |
| SVR (RBF) | 78.24 | 279.89 | 1.180 | 1.019 | 0.00537 | 0.00809 |
| XGBoost | 83.95 | 300.63 | **1.038** | 0.523 | **0.00491** | 0.00678 |
| Ridge | 25,917 | 93,445 | 13.131 | 43.925 | 0.633 | 2.271 |
| Lasso | 25,918 | 93,445 | 13.131 | 43.925 | 0.633 | 2.271 |

All metrics averaged across 13 folds. Scores reported on variance (not log) scale.

### Key Findings

- **GARCH(1,1) is the best overall model** on the primary QLIKE metric, reinforcing that
  volatility clustering is well-captured by a simple ARCH recursion.
- **XGBoost achieves the lowest MSE-log** (1.038), beating GARCH (1.338) and all others
  on that metric.
- **SVR (RBF) is the best ML model on QLIKE** (78.24), though it still trails GARCH.
- **Ridge and Lasso fail spectacularly** (QLIKE > 25,000), driven by QLIKE's extreme
  sensitivity to underprediction when predictions in log-space go far negative in tail folds.
- **ML did not beat classical baselines** on QLIKE — in line with the broader empirical
  finance literature on the robustness of GARCH-family models.

### Figures

| Figure | Description |
|---|---|
| `results/figures/qlike_per_fold.png` | Per-fold QLIKE for all 6 models (2014–2026) |
| `results/figures/predicted_vs_actual_covid.png` | Fold 7 (2020) COVID stress test — all model predictions vs actual YZ vol |

---

## Project Structure

```
vol_forecasting/
├── data/
│   ├── raw/
│   │   └── spy_ohlc.csv                  # 4,619 rows of SPY OHLCV
│   └── processed/
│       └── spy_features.parquet          # Engineered feature matrix (date index)
├── src/
│   ├── __init__.py                       # Global SEED = 22
│   ├── volatility_estimators.py          # YZ, GK, Parkinson, Rogers-Satchell
│   ├── features.py                       # 22-feature pipeline + no-lookahead assertion
│   ├── models.py                         # 6 unified model wrappers
│   ├── validation.py                     # Walk-forward loop + results summary
│   └── metrics.py                        # QLIKE, MSE-log, MAE-vol, Diebold-Mariano
├── scripts/
│   ├── convert_json_to_csv.py            # Godel Terminal JSON → data/raw/spy_ohlc.csv
│   ├── smoke_test.py                     # numpy/pandas-only sanity pass (no ML deps)
│   └── run_pipeline.py                   # Full end-to-end pipeline driver
├── results/
│   ├── predictions/                      # 78 parquets: {model}_fold{n}.parquet
│   ├── figures/
│   │   ├── qlike_per_fold.png
│   │   └── predicted_vs_actual_covid.png
│   ├── per_fold_losses.csv               # 78 rows × 8 cols (fold × model × metrics)
│   └── results_summary.csv              # 6-model × 3-metric aggregated summary
├── Historical_Prices_ETC_SPY_2026-04-28.json  # Source data (Godel Terminal export)
├── requirements.txt
├── vol_forecasting_spec.md              # Full project specification
└── README.md
```

---

## Data Pipeline

### Source

- **Asset:** SPY (SPDR S&P 500 ETF Trust)
- **Source:** Godel Terminal daily OHLCV export
- **File:** `Historical_Prices_ETC_SPY_2026-04-28.json`
- **Date range:** 2008-01-02 to 2026-04-28

### JSON → CSV Conversion (`scripts/convert_json_to_csv.py`)

Parses the Godel Terminal JSON format, applies the following transformations:
- UTC date to local timezone conversion
- Deduplication by date (keep last record)
- Validation: monotonic dates, positive OHLC values, high ≥ low
- Output: `data/raw/spy_ohlc.csv` (4,619 rows: date, open, high, low, close, volume)

### Feature Engineering (`src/features.py`)

Feature matrix saved as `data/processed/spy_features.parquet` with a date index.

**Target variable:**
- Single-day Yang-Zhang variance for each date
- Shifted forward by 1 day (`shift(-1)`) — predicting *tomorrow's* variance using *yesterday and earlier* data
- Stored as `log(YZ variance)` floored at `1e-10` to avoid `log(0)`

---

## Volatility Estimators

All estimators are implemented in `src/volatility_estimators.py` and return **variance** (not
volatility). Take `sqrt()` for volatility.

| Estimator | Formula | Property |
|---|---|---|
| **Yang-Zhang (YZ)** | σ²_overnight + 0.34·σ²_OC + 0.66·σ²_RS | Minimum variance, handles overnight jumps |
| **Garman-Klass (GK)** | 0.5·ln(H/L)² − (2ln2−1)·ln(C/O)² | Efficient range estimator |
| **Parkinson** | ln(H/L)² / (4·ln2) | Range-based, simple |
| **Rogers-Satchell (RS)** | ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) | Drift-independent |

Yang-Zhang supports both single-day and rolling-window computation. The rolling variant is
used internally for feature construction.

---

## Feature Engineering

### 22 Features (all causal — `t-1` or earlier)

| Category | Feature Names | Count |
|---|---|---|
| Lagged YZ volatility | `rv_lag1`, `rv_lag5_mean`, `rv_lag10_mean`, `rv_lag22_mean`, `rv_lag66_mean` | 5 |
| Lagged log returns | `ret_lag1`, `ret_lag5_mean` | 2 |
| Lagged absolute returns | `abs_ret_lag1`, `abs_ret_lag5_mean` | 2 |
| Leverage effect | `signed_ret_x_vol_lag1` (return × vol, lag 1) | 1 |
| Alt vol estimators | `parkinson_lag1`, `garman_klass_lag1` | 2 |
| Return decomposition | `overnight_ret_lag1`, `intraday_ret_lag1` | 2 |
| Higher moments (22-day) | `ret_skew_22`, `ret_kurt_22` | 2 |
| High-low range | `hl_range_lag1` | 1 |
| Vol-of-vol | `vov_22` (22-day rolling std of daily YZ vol) | 1 |
| Day-of-week dummies | `dow_mon`, `dow_tue`, `dow_wed`, `dow_thu` (Fri = reference) | 4 |
| **Total** | | **22** |

### No-Lookahead Enforcement

Every feature is constructed with `.shift(1)` or earlier relative to the prediction date.
The function `_assert_no_lookahead()` in `src/features.py` validates at mid-sample that
feature values can only be explained by data strictly before the target date. The pipeline
will raise `AssertionError` if this contract is violated.

---

## Models

All models implement a unified interface:

```python
model.fit(X_train, y_train)    # y in log-variance space
model.predict(X_test)          # returns log-variance predictions (np.ndarray)
model.best_params_             # best hyperparams (dict), populated after fit
model.feature_importance_      # dict or None
```

### HAR-RV (Baseline)

Heterogeneous Autoregressive model on Realized Volatility (Corsi, 2009).

- **Features:** 3 only — `rv_lag1`, `rv_lag5_mean`, `rv_lag22_mean`
- **Algorithm:** OLS (`sklearn.linear_model.LinearRegression`)
- **Tuning:** None

### GARCH(1,1) (Baseline)

Generalized Autoregressive Conditional Heteroskedasticity (Bollerslev, 1986).

- **Input:** Log returns (not feature matrix — uses raw return series)
- **Algorithm:** `arch.arch_model(returns, vol='GARCH', p=1, q=1, mean='Constant')`, MLE fit
- **Forecast:** Fitted once per fold on full training returns; 1-step-ahead variance
  rolled forward through the test window without parameter refits
- **Tuning:** None

### Ridge Regression

- **Features:** All 22, `StandardScaler` fit on training set only
- **Hyperparameter search:** `alpha` ∈ `logspace(-3, 3, 13)`, `GridSearchCV`
- **CV:** `TimeSeriesSplit(n_splits=5)` inside training window
- **Scoring:** `neg_mean_squared_error`

### Lasso Regression

- **Features:** All 22, `StandardScaler` fit on training set only
- **Hyperparameter search:** Same `alpha` grid as Ridge, `GridSearchCV`
- **CV:** `TimeSeriesSplit(n_splits=5)` inside training window
- **Note:** `max_iter=10000`, `random_state=SEED`

### SVR (RBF Kernel)

- **Features:** All 22, `StandardScaler` fit on training set only
- **Hyperparameter search:** `RandomizedSearchCV` with 20 iterations
  - `C`: loguniform(0.1, 100)
  - `gamma`: loguniform(1e-4, 1)
  - `epsilon`: uniform(0.001, 0.1)
- **CV:** `TimeSeriesSplit(n_splits=5)` inside training window

### XGBoost

- **Features:** All 22, unscaled (tree models are scale-invariant)
- **Hyperparameter search:** `RandomizedSearchCV` with 30 iterations
  - `n_estimators`: [100, 200, 400, 800]
  - `max_depth`: [3, 4, 5, 6, 8]
  - `learning_rate`: loguniform(0.01, 0.3)
  - `subsample`, `colsample_bytree`: uniform(0.6, 1.0)
  - `reg_alpha`, `reg_lambda`: loguniform(1e-3, 1)
- **Early stopping:** 50 rounds on an 85/15 train/validation split
- **Feature importance:** `gain`-based, extracted after final fit

---

## Walk-Forward Validation

**Protocol:** Expanding-window, annual test folds. Implemented in `src/validation.py`.

- **Initial training window:** 2008-01-01 through 2013-12-31 (~1,500 trading days)
- **Test folds:** Annual from 2014 through April 2026 (13 folds total)
- **Total out-of-sample observations:** ~3,000 per model

| Fold | Train Window | Test Window | approx. n_train | approx. n_test |
|---|---|---|---|---|
| 1 | 2008-01-01 .. 2013-12-31 | 2014 | 1,500 | 253 |
| 2 | 2008-01-01 .. 2014-12-31 | 2015 | 1,753 | 252 |
| 3 | 2008-01-01 .. 2015-12-31 | 2016 | 2,005 | 252 |
| 4 | 2008-01-01 .. 2016-12-31 | 2017 | 2,257 | 251 |
| 5 | 2008-01-01 .. 2017-12-31 | 2018 | 2,508 | 251 |
| 6 | 2008-01-01 .. 2018-12-31 | 2019 | 2,759 | 252 |
| 7 | 2008-01-01 .. 2019-12-31 | 2020 | 3,011 | 253 |
| 8 | 2008-01-01 .. 2020-12-31 | 2021 | 3,264 | 252 |
| 9 | 2008-01-01 .. 2021-12-31 | 2022 | 3,516 | 251 |
| 10 | 2008-01-01 .. 2022-12-31 | 2023 | 3,767 | 250 |
| 11 | 2008-01-01 .. 2023-12-31 | 2024 | 4,017 | 252 |
| 12 | 2008-01-01 .. 2024-12-31 | 2025 | 4,269 | 252 |
| 13 | 2008-01-01 .. 2025-12-31 | Jan–Apr 2026 | 4,521 | ~80 |

**Critical design decisions:**
- Hyperparameter tuning is nested entirely within each fold's training window — the test fold
  is never touched during model selection
- All nested cross-validation uses `TimeSeriesSplit` (no shuffling)
- Predictions are stored per fold in `results/predictions/{model}_fold{n}.parquet` for
  full forensic access

---

## Loss Functions

All metrics operate on **variance** (not log-variance, not volatility). Implemented in `src/metrics.py`.

Predictions are clipped to `[-25, 5]` in log-space before exponentiation as a numerical
safety guard against extreme outliers in early folds.

### QLIKE (Primary Metric)

```
QLIKE = mean( σ²_true / σ²_pred − log(σ²_true / σ²_pred) − 1 )
```

The standard loss function for volatility forecast evaluation (Patton, 2011). Robust to
noisy volatility proxies. Severely penalizes underprediction, which is why Ridge/Lasso
fail when they predict near-zero variance in tail regimes.

### MSE on Log-Variance (Secondary)

```
MSE_log = mean( (log σ²_true − log σ²_pred)² )
```

Standard regression loss in log-space. More symmetric than QLIKE; used to compare models
that differ in tail behavior.

### MAE on Volatility (Tertiary)

```
MAE_vol = mean( |σ_true − σ_pred| )
```

Most interpretable metric — units are annualized vol points. Useful for translating
forecast accuracy into practical sizing terms.

### Diebold-Mariano Test

Implemented in `src/metrics.py`. Computes per-observation loss differences with
Newey-West long-run variance correction and returns a DM statistic and two-sided p-value.
Available for pairwise model comparisons.

---

## Reproducing Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Smoke test — no ML packages required

Validates the data pipeline, feature engineering, fold structure, and metrics using only
`numpy`, `pandas`, and `pyarrow`. Safe to run in any environment.

```bash
python scripts/smoke_test.py
```

### 3. Full pipeline

Runs all 6 models across all 13 folds. Requires `scikit-learn`, `xgboost`, and `arch`.
Estimated runtime: 30–90 minutes depending on hardware.

```bash
python scripts/run_pipeline.py
```

**Pipeline steps (in order):**
1. Convert `Historical_Prices_ETC_SPY_2026-04-28.json` → `data/raw/spy_ohlc.csv`
2. Build feature matrix → `data/processed/spy_features.parquet`
3. Walk-forward validation loop (6 models × 13 folds)
4. Write per-fold predictions → `results/predictions/{model}_fold{n}.parquet`
5. Aggregate losses → `results/per_fold_losses.csv`, `results/results_summary.csv`
6. Render figures → `results/figures/qlike_per_fold.png`, `results/figures/predicted_vs_actual_covid.png`

All random state is seeded via `SEED = 22` in `src/__init__.py`. Results should be
bit-for-bit reproducible across runs on the same hardware/library versions.

---

## Dependencies

```
numpy>=1.26
pandas>=2.0
scikit-learn>=1.4
xgboost>=2.0
arch>=6.3
scipy>=1.11
matplotlib>=3.8
seaborn>=0.13
pyarrow>=14.0
jupyter>=1.0
```

---

## References

- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.*
  Journal of Econometrics, 31(3), 307–327.

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.*
  Journal of Financial Econometrics, 7(2), 174–196.

- Garman, M. B., & Klass, M. J. (1980). *On the Estimation of Security Price Volatilities
  from Historical Data.* Journal of Business, 53(1), 67–78.

- Parkinson, M. (1980). *The Extreme Value Method for Estimating the Variance of the Rate
  of Return.* Journal of Business, 53(1), 61–65.

- Patton, A. J. (2011). *Volatility Forecast Comparison Using Imperfect Volatility Proxies.*
  Journal of Econometrics, 160(1), 246–256.

- Rogers, L. C. G., & Satchell, S. E. (1991). *Estimating Variance from High, Low and
  Closing Prices.* The Annals of Applied Probability, 1(4), 504–512.

- Yang, D., & Zhang, Q. (2000). *Drift-Independent Volatility Estimation Based on High,
  Low, Open, and Close Prices.* Journal of Business, 73(3), 477–491.
