"""Loss functions for volatility forecast evaluation.

All metrics operate on *variance* (not vol, not log-variance). The runner
converts predictions from log-variance to variance before scoring.
"""
from __future__ import annotations

import numpy as np

VAR_FLOOR = 1e-12


def _as_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a


def qlike(var_true, var_pred) -> float:
    """QLIKE loss: mean(σ²_true/σ²_pred - log(σ²_true/σ²_pred) - 1).

    Robust to noisy volatility proxies (Patton, 2011). Lower is better.
    """
    vt = np.maximum(_as_array(var_true), VAR_FLOOR)
    vp = np.maximum(_as_array(var_pred), VAR_FLOOR)
    r = vt / vp
    return float(np.mean(r - np.log(r) - 1.0))


def mse_log_variance(var_true, var_pred) -> float:
    """MSE on log-variance. Standard regression loss in log space."""
    vt = np.maximum(_as_array(var_true), VAR_FLOOR)
    vp = np.maximum(_as_array(var_pred), VAR_FLOOR)
    return float(np.mean((np.log(vt) - np.log(vp)) ** 2))


def mae_volatility(var_true, var_pred) -> float:
    """MAE on volatility (sqrt of variance). Most interpretable."""
    vt = np.maximum(_as_array(var_true), VAR_FLOOR)
    vp = np.maximum(_as_array(var_pred), VAR_FLOOR)
    return float(np.mean(np.abs(np.sqrt(vt) - np.sqrt(vp))))


def all_metrics(var_true, var_pred) -> dict[str, float]:
    return {
        "qlike": qlike(var_true, var_pred),
        "mse_log": mse_log_variance(var_true, var_pred),
        "mae_vol": mae_volatility(var_true, var_pred),
    }


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test on two per-observation loss series.

    Returns (dm_stat, two_sided_p_value). Positive dm_stat means loss_a > loss_b
    on average (i.e. model B is better).
    """
    la = _as_array(loss_a)
    lb = _as_array(loss_b)
    d = la - lb
    n = d.size
    if n < 10:
        return float("nan"), float("nan")
    mean_d = d.mean()
    # Newey-West long-run variance with lag h-1
    gamma0 = np.mean((d - mean_d) ** 2)
    lrv = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        lrv += 2.0 * (1.0 - lag / h) * cov
    lrv = max(lrv, 1e-16)
    dm_stat = mean_d / np.sqrt(lrv / n)
    # Two-sided p-value from the standard normal (harmless approximation for
    # moderate n; scipy.stats.t is slightly more conservative)
    from math import erf, sqrt
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0))))
    return float(dm_stat), float(p_value)
