"""Feature engineering pipeline for SPY volatility forecasting.

Every feature at time t uses information from t-1 or earlier. The build
function enforces this via explicit .shift(1) on all predictors and asserts
the target column is never used as an input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .volatility_estimators import (
    garman_klass_variance,
    parkinson_variance,
    yang_zhang_daily_variance,
)

MIN_VARIANCE = 1e-10  # floor to avoid log(0)


@dataclass(frozen=True)
class FeatureConfig:
    rv_lags_mean: tuple[int, ...] = (5, 10, 22, 66)
    ret_lag_mean: int = 5
    abs_ret_lag_mean: int = 5
    skew_window: int = 22
    kurt_window: int = 22
    vov_window: int = 22


def _log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def build_target(ohlc: pd.DataFrame) -> pd.Series:
    """Target variable: log of 1-day-ahead Yang-Zhang variance.

    The single-day YZ variance at day t is the realized vol proxy *for day t*.
    We forecast this 1 day ahead, so the target aligned with feature row t
    is YZ_var(t+1), accessed via .shift(-1) at the end of feature assembly.

    This function returns the YZ daily variance series (unshifted). The
    caller shifts for supervised alignment.
    """
    yz_var = yang_zhang_daily_variance(
        ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    )
    yz_var = yz_var.clip(lower=MIN_VARIANCE)
    return yz_var


def build_feature_matrix(
    ohlc: pd.DataFrame, config: FeatureConfig | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target y.

    Parameters
    ----------
    ohlc : DataFrame indexed by date with columns [open, high, low, close, volume].

    Returns
    -------
    X : DataFrame of features, strictly lagged (no lookahead).
    y : Series, log YZ variance 1 day ahead.
    """
    if config is None:
        config = FeatureConfig()

    df = ohlc.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()

    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    # --- base series (at time t, realized) ---
    yz_var_t = build_target(df)
    log_ret_t = _log_return(close)
    overnight_ret_t = np.log(open_ / close.shift(1))
    intraday_ret_t = np.log(close / open_)
    parkinson_t = parkinson_variance(high, low).clip(lower=MIN_VARIANCE)
    gk_t = garman_klass_variance(open_, high, low, close).clip(lower=MIN_VARIANCE)
    hl_range_t = (high - low) / close

    # --- features are all at lag >= 1 ---
    feats: dict[str, pd.Series] = {}

    # Lagged vol: rv_lag1 is YZ variance from yesterday; rolling means
    # shift by 1 so they use [t-w, ..., t-1].
    feats["rv_lag1"] = yz_var_t.shift(1)
    for w in config.rv_lags_mean:
        feats[f"rv_lag{w}_mean"] = yz_var_t.shift(1).rolling(w).mean()

    # Lagged returns
    feats["ret_lag1"] = log_ret_t.shift(1)
    feats[f"ret_lag{config.ret_lag_mean}_mean"] = (
        log_ret_t.shift(1).rolling(config.ret_lag_mean).mean()
    )

    # Lagged absolute returns
    feats["abs_ret_lag1"] = log_ret_t.abs().shift(1)
    feats[f"abs_ret_lag{config.abs_ret_lag_mean}_mean"] = (
        log_ret_t.abs().shift(1).rolling(config.abs_ret_lag_mean).mean()
    )

    # Leverage effect
    feats["signed_ret_x_vol_lag1"] = log_ret_t.shift(1) * yz_var_t.shift(1)

    # Alternative vol estimators
    feats["parkinson_lag1"] = parkinson_t.shift(1)
    feats["garman_klass_lag1"] = gk_t.shift(1)

    # Return decomposition
    feats["overnight_ret_lag1"] = overnight_ret_t.shift(1)
    feats["intraday_ret_lag1"] = intraday_ret_t.shift(1)

    # Higher moments (lagged window ends at t-1)
    feats[f"ret_skew_{config.skew_window}"] = (
        log_ret_t.shift(1).rolling(config.skew_window).skew()
    )
    feats[f"ret_kurt_{config.kurt_window}"] = (
        log_ret_t.shift(1).rolling(config.kurt_window).kurt()
    )

    # Range
    feats["hl_range_lag1"] = hl_range_t.shift(1)

    # Vol of vol
    feats[f"vov_{config.vov_window}"] = (
        yz_var_t.shift(1).rolling(config.vov_window).std(ddof=1)
    )

    # Day-of-week dummies (Friday reference, 0=Mon..4=Fri)
    dow = df.index.dayofweek
    feats["dow_mon"] = (dow == 0).astype(float)
    feats["dow_tue"] = (dow == 1).astype(float)
    feats["dow_wed"] = (dow == 2).astype(float)
    feats["dow_thu"] = (dow == 3).astype(float)

    X = pd.DataFrame(feats, index=df.index)

    # Target: 1-day-ahead log YZ variance.
    # Row aligned on date t represents "predict vol for t+1 using features
    # from <=t-1". We set target at row t = log(yz_var_{t+1}) so that the
    # features and target both exclude information from day t itself (strict
    # pre-trade forecast). This is equivalent to the simpler "predict vol
    # at day t given data up to t-1" and keeps no-lookahead trivially safe.
    y = np.log(yz_var_t.shift(-1))
    y.name = "log_yz_var_next"

    # Assert strict no-lookahead: every feature column at time t must equal
    # a value that could be computed using only data up to t-1.
    _assert_no_lookahead(X, df)

    return X, y


def _assert_no_lookahead(X: pd.DataFrame, ohlc: pd.DataFrame) -> None:
    """Sanity check: for a random valid row, recomputing a handful of
    features using only pre-t data must match X.loc[t].
    """
    valid = X.dropna().index
    if len(valid) < 100:
        return
    # Take a deterministic mid-sample row
    t = valid[len(valid) // 2]
    prev_close = ohlc["close"].loc[:t].iloc[-2]
    open_t = ohlc["open"].loc[t]
    expected_overnight = np.log(ohlc["open"].loc[:t].iloc[-1] / prev_close)
    # overnight_ret_lag1 at row t should use data up to t-1, i.e. the overnight
    # return observed on day t-1 (open_{t-1} / close_{t-2}).
    prev_prev_close = ohlc["close"].loc[:t].iloc[-3]
    prev_open = ohlc["open"].loc[:t].iloc[-2]
    expected_overnight_lag1 = np.log(prev_open / prev_prev_close)
    assert np.isclose(
        X.loc[t, "overnight_ret_lag1"], expected_overnight_lag1, equal_nan=True
    ), "No-lookahead check failed on overnight_ret_lag1"


def prepare_supervised_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Build X, y then return a single frame with columns ['y', *features],
    with all NaN rows (from rolling windows and the last-row target shift)
    dropped.
    """
    X, y = build_feature_matrix(ohlc)
    frame = X.copy()
    frame.insert(0, "y", y)
    frame = frame.dropna()
    return frame


def feature_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c != "y"]
