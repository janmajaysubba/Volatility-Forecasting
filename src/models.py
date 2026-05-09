"""Unified model wrappers for walk-forward validation.

Every wrapper implements:
    fit(X_train, y_train, dates_train=None) -> self
    predict(X_test) -> np.ndarray
    best_params_ -> dict (populated after fit for tuned models)
    feature_importance_ -> dict[str, float] | None

Targets are in log-variance space. The wrappers do not exponentiate —
that's the runner's job (see src/validation.py).

GARCH is a special case: it ignores the engineered feature matrix and
instead consumes a returns series that the runner passes in via
dates_train / the auxiliary `returns` kwarg on fit/predict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import SEED


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseModel:
    name: str = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def best_params_(self) -> Dict[str, Any]:
        return getattr(self, "_best_params", {})

    @property
    def feature_importance_(self) -> Optional[Dict[str, float]]:
        return getattr(self, "_feature_importance", None)


# ---------------------------------------------------------------------------
# HAR-RV (Corsi 2009)
# ---------------------------------------------------------------------------

HAR_FEATURES = ["rv_lag1", "rv_lag5_mean", "rv_lag22_mean"]


class HARRVModel(BaseModel):
    name = "har_rv"

    def __init__(self) -> None:
        from sklearn.linear_model import LinearRegression

        self._model = LinearRegression()
        self._features = HAR_FEATURES

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "HARRVModel":
        missing = [f for f in self._features if f not in X.columns]
        if missing:
            raise ValueError(f"HAR-RV missing features: {missing}")
        self._model.fit(X[self._features].values, y.values)
        self._best_params = {}
        coefs = dict(zip(self._features, self._model.coef_))
        coefs["intercept"] = float(self._model.intercept_)
        self._feature_importance = coefs
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._model.predict(X[self._features].values)


# ---------------------------------------------------------------------------
# GARCH(1,1) (Bollerslev 1986)
# ---------------------------------------------------------------------------

class GARCH11Model(BaseModel):
    """GARCH(1,1) on log returns.

    Differs from the other models: it ignores the engineered feature matrix
    and instead fits on a returns series. To avoid refitting 252 times per
    fold (prohibitively slow), we fit once per test *fold* on all returns up
    to the fold start, then roll forward one-step-ahead variance forecasts
    through the test window using the fitted parameters (i.e. no parameter
    refits within the fold — standard approach in the forecasting
    literature).
    """
    name = "garch11"

    def __init__(self) -> None:
        self._fitted = None
        self._best_params = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: Optional[pd.Series] = None,
        **kwargs,
    ) -> "GARCH11Model":
        from arch import arch_model

        if returns is None:
            raise ValueError("GARCH11Model.fit requires `returns=` kwarg")
        # arch works in % returns to keep optimizer well-scaled
        r = (returns.dropna() * 100.0).astype(float)
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        self._fitted = am.fit(disp="off", show_warning=False)
        self._returns_scale = 100.0
        self._train_returns = r
        p = self._fitted.params.to_dict()
        self._best_params = {
            "omega": p.get("omega"),
            "alpha[1]": p.get("alpha[1]"),
            "beta[1]": p.get("beta[1]"),
        }
        return self

    def predict(
        self,
        X: pd.DataFrame,
        test_returns: Optional[pd.Series] = None,
        **kwargs,
    ) -> np.ndarray:
        """One-step-ahead conditional variance over the test window.

        For each test date, we roll the GARCH recursion forward using the
        most recent realized return. The returned variance is in the
        *original* return scale (not %), and we return log-variance to match
        the other models' target.
        """
        if self._fitted is None:
            raise RuntimeError("GARCH11Model: fit before predict")
        if test_returns is None:
            raise ValueError("GARCH11Model.predict requires `test_returns=` kwarg")

        p = self._fitted.params
        omega = float(p["omega"])
        alpha = float(p["alpha[1]"])
        beta = float(p["beta[1]"])
        mu = float(p.get("mu", 0.0))

        # Starting conditional variance: last in-sample h_t (in %²)
        h_prev = float(self._fitted.conditional_volatility.iloc[-1] ** 2)
        try:
            last_resid = float(self._fitted.resid.dropna().iloc[-1])
        except Exception:
            last_resid = float(self._train_returns.dropna().iloc[-1] - mu)

        test_r_pct = (test_returns * 100.0).astype(float).values
        n = len(X)
        preds_var_pct = np.empty(n)

        # One-step-ahead variance using info up to t-1
        for i in range(n):
            h_t = omega + alpha * last_resid ** 2 + beta * h_prev
            preds_var_pct[i] = h_t
            # roll state forward using realized return at t (becomes "t-1" next step)
            r_t = test_r_pct[i] if i < len(test_r_pct) else 0.0
            last_resid = r_t - mu
            h_prev = h_t

        # Convert back from %² variance to raw variance
        preds_var = preds_var_pct / (self._returns_scale ** 2)
        return np.log(np.maximum(preds_var, 1e-12))


# ---------------------------------------------------------------------------
# Linear shrinkage models
# ---------------------------------------------------------------------------

class _ScaledLinearModel(BaseModel):
    """Shared scaffolding for Ridge / Lasso with TimeSeriesSplit alpha tuning."""

    alphas = np.logspace(-3, 3, 13)

    def __init__(self) -> None:
        self._pipeline = None
        self._best_params = {}
        self._feature_importance = None

    def _make_estimator(self):  # overridden
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "_ScaledLinearModel":
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", self._make_estimator()),
            ]
        )
        param_grid = {"model__alpha": self.alphas}
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        gs.fit(X.values, y.values)
        self._pipeline = gs.best_estimator_
        self._best_params = {"alpha": float(gs.best_params_["model__alpha"])}
        coefs = self._pipeline.named_steps["model"].coef_
        self._feature_importance = dict(zip(X.columns, map(float, coefs)))
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._pipeline.predict(X.values)


class RidgeModel(_ScaledLinearModel):
    name = "ridge"

    def _make_estimator(self):
        from sklearn.linear_model import Ridge

        return Ridge(random_state=SEED)


class LassoModel(_ScaledLinearModel):
    name = "lasso"

    def _make_estimator(self):
        from sklearn.linear_model import Lasso

        return Lasso(max_iter=10000, random_state=SEED)


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------

class SVRModel(BaseModel):
    name = "svr"

    def __init__(self) -> None:
        self._pipeline = None
        self._best_params = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "SVRModel":
        from scipy.stats import loguniform, uniform
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]
        )
        param_dist = {
            "model__C": loguniform(0.1, 100),
            "model__gamma": loguniform(1e-4, 1),
            "model__epsilon": uniform(0.001, 0.099),
        }
        tscv = TimeSeriesSplit(n_splits=5)
        rs = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring="neg_mean_squared_error",
            random_state=SEED,
            n_jobs=-1,
        )
        rs.fit(X.values, y.values)
        self._pipeline = rs.best_estimator_
        self._best_params = {
            k.replace("model__", ""): v for k, v in rs.best_params_.items()
        }
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._pipeline.predict(X.values)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self) -> None:
        self._model = None
        self._best_params = {}
        self._feature_importance = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "XGBoostModel":
        from scipy.stats import loguniform, uniform
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        from xgboost import XGBRegressor

        self._columns = list(X.columns)

        base = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        param_dist = {
            "n_estimators": [100, 200, 400, 800],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),       # U[0.6, 1.0]
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": loguniform(1e-3, 1),
            "reg_lambda": loguniform(1e-3, 1),
        }
        tscv = TimeSeriesSplit(n_splits=5)
        rs = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=30,
            cv=tscv,
            scoring="neg_mean_squared_error",
            random_state=SEED,
            n_jobs=-1,
        )
        rs.fit(X.values, y.values)
        best = rs.best_params_

        # Refit final model with early stopping on a held-out tail of training.
        n = len(X)
        split = int(n * 0.85)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        final = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50,
            **best,
        )
        final.fit(
            X_tr.values, y_tr.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )
        self._model = final
        self._best_params = {k: _maybe_float(v) for k, v in best.items()}
        importances = final.feature_importances_
        self._feature_importance = dict(zip(self._columns, map(float, importances)))
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._model.predict(X[self._columns].values)


def _maybe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_models() -> dict[str, BaseModel]:
    """Factory — returns a fresh model dict for each fold."""
    return {
        "har_rv": HARRVModel(),
        "garch11": GARCH11Model(),
        "ridge": RidgeModel(),
        "lasso": LassoModel(),
        "svr": SVRModel(),
        "xgboost": XGBoostModel(),
    }
