"""No-dependency smoke test: JSON -> CSV -> feature matrix -> metrics.

Runs only the parts of the pipeline that need numpy + pandas + pyarrow
(no sklearn/xgboost/arch). Useful while those packages are not yet
installed.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_feature_matrix
from src.metrics import all_metrics, qlike
from src.volatility_estimators import (
    garman_klass_variance,
    parkinson_variance,
    rogers_satchell_variance,
    yang_zhang_daily_variance,
    yang_zhang_rolling_variance,
)
from src.validation import make_annual_folds

RAW_CSV = ROOT / "data" / "raw" / "spy_ohlc.csv"


def _load_ohlc() -> pd.DataFrame:
    if not RAW_CSV.exists():
        print(f"Raw CSV missing; running converter...")
        from scripts.convert_json_to_csv import main

        main()
    df = pd.read_csv(RAW_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    ohlc = _load_ohlc()
    print(f"OHLC rows: {len(ohlc):,}  range: {ohlc['date'].min().date()}..{ohlc['date'].max().date()}")

    # --- volatility estimators ---
    idx = ohlc.set_index("date")
    pk = parkinson_variance(idx["high"], idx["low"])
    gk = garman_klass_variance(idx["open"], idx["high"], idx["low"], idx["close"])
    rs = rogers_satchell_variance(idx["open"], idx["high"], idx["low"], idx["close"])
    yz = yang_zhang_daily_variance(idx["open"], idx["high"], idx["low"], idx["close"])
    yz_roll = yang_zhang_rolling_variance(
        idx["open"], idx["high"], idx["low"], idx["close"], window=22
    )

    def _ann(var_series: pd.Series) -> float:
        return float(np.sqrt(var_series.dropna().mean() * 252))

    print("\nAnnualized mean volatility (full sample):")
    print(f"  Parkinson         {_ann(pk):.4f}")
    print(f"  Garman-Klass      {_ann(gk):.4f}")
    print(f"  Rogers-Satchell   {_ann(rs):.4f}")
    print(f"  Yang-Zhang (1d)   {_ann(yz):.4f}")
    print(f"  YZ rolling-22     {_ann(yz_roll):.4f}")

    # --- feature matrix ---
    X, y = build_feature_matrix(ohlc)
    frame = X.copy()
    frame.insert(0, "y", y)
    frame = frame.dropna()
    print(f"\nFeature matrix shape: {frame.shape}")
    print(f"  feature columns ({len(X.columns)}): {list(X.columns)}")
    print(f"  target NaNs before dropna: {y.isna().sum()}")
    print(f"  usable date range: {frame.index.min().date()}..{frame.index.max().date()}")

    # Correlations with target — sanity check
    print("\nTop-5 features by |corr(y, feature)|:")
    corrs = frame.corr()["y"].drop("y").abs().sort_values(ascending=False).head(5)
    print(corrs.to_string())

    # --- folds ---
    folds = make_annual_folds(
        start_date=str(frame.index.min().date()),
        last_date=str(frame.index.max().date()),
    )
    print(f"\nGenerated {len(folds)} walk-forward folds:")
    for f in folds:
        n_train = frame.loc[f.train_start : f.train_end].shape[0]
        n_test = frame.loc[f.test_start : f.test_end].shape[0]
        print(
            f"  fold {f.index:02d}: train {f.train_start.date()}..{f.train_end.date()} ({n_train:4d} rows)  "
            f"test {f.test_start.date()}..{f.test_end.date()} ({n_test:4d} rows)"
        )

    # --- metrics sanity: perfect prediction gives QLIKE=0 ---
    var_true = np.exp(frame["y"].values)
    print("\nMetrics sanity:")
    print(f"  QLIKE(perfect)        = {qlike(var_true, var_true):.6e}")
    print(f"  QLIKE(2x overestimate)= {qlike(var_true, var_true * 2):.4f}")
    print(f"  QLIKE(2x underest.)   = {qlike(var_true, var_true / 2):.4f}")
    # Naive RV baseline: predict tomorrow's variance = today's (rv_lag1 is
    # already the YZ variance at t-1, so use it directly — no exp).
    naive_pred = frame["rv_lag1"].values
    m = all_metrics(var_true, naive_pred)
    print(f"  Naive 'rv_lag1' as predictor (full sample): {m}")


if __name__ == "__main__":
    main()
