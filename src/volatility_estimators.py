"""Range-based and OHLC volatility estimators.

All functions return *variance* series (not volatility). Take sqrt for vol.
Inputs are pandas Series aligned on the same DatetimeIndex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YZ_K_DEFAULT = 0.34


def _log(x: pd.Series) -> pd.Series:
    return np.log(x)


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) single-day variance from high-low range."""
    ln_hl = _log(high / low)
    return (ln_hl ** 2) / (4.0 * np.log(2.0))


def garman_klass_variance(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Garman-Klass (1980) single-day variance."""
    ln_hl = _log(high / low)
    ln_co = _log(close / open_)
    return 0.5 * ln_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * ln_co ** 2


def rogers_satchell_variance(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Rogers-Satchell (1991) single-day variance, drift-independent."""
    return (
        _log(high / close) * _log(high / open_)
        + _log(low / close) * _log(low / open_)
    )


def yang_zhang_daily_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: float = YZ_K_DEFAULT,
) -> pd.Series:
    """Single-day Yang-Zhang (2000) variance.

    Uses overnight + k * open-to-close + (1-k) * Rogers-Satchell.
    Overnight uses previous close, so the first observation is NaN.
    """
    prev_close = close.shift(1)
    sigma2_overnight = _log(open_ / prev_close) ** 2
    sigma2_open_close = _log(close / open_) ** 2
    sigma2_rs = rogers_satchell_variance(open_, high, low, close)
    return sigma2_overnight + k * sigma2_open_close + (1.0 - k) * sigma2_rs


def yang_zhang_rolling_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 22,
    k: float | None = None,
) -> pd.Series:
    """Rolling Yang-Zhang variance (panel / multi-day formulation).

    σ²_YZ = σ²_overnight + k · σ²_open-to-close + (1-k) · σ²_RS

    where σ²_overnight and σ²_open-to-close are rolling sample variances
    (demeaned within the window) and σ²_RS is the rolling mean of the
    Rogers-Satchell single-day variance.
    """
    prev_close = close.shift(1)
    overnight_ret = _log(open_ / prev_close)
    open_close_ret = _log(close / open_)

    var_overnight = overnight_ret.rolling(window).var(ddof=1)
    var_open_close = open_close_ret.rolling(window).var(ddof=1)
    rs_daily = rogers_satchell_variance(open_, high, low, close)
    var_rs = rs_daily.rolling(window).mean()

    if k is None:
        # Yang-Zhang optimal k for the rolling estimator
        alpha = 1.34
        k = (alpha - 1.0) / (alpha + (window + 1.0) / (window - 1.0))

    return var_overnight + k * var_open_close + (1.0 - k) * var_rs


def close_to_close_variance(close: pd.Series, window: int = 22) -> pd.Series:
    """Classical close-to-close rolling variance of log returns."""
    log_ret = _log(close / close.shift(1))
    return log_ret.rolling(window).var(ddof=1)
