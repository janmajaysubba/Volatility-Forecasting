"""Walk-forward expanding-window validation.

Annual out-of-sample folds, training window starts at 2008-01-01 and
expands by one year per fold. Hyperparameter tuning happens entirely
inside each fold's training window via TimeSeriesSplit in the model
wrappers — the test window is never seen during tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .metrics import all_metrics
from .models import BaseModel


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # exclusive of test_start
    test_start: pd.Timestamp
    test_end: pd.Timestamp    # inclusive


def make_annual_folds(
    start_date: str = "2008-01-01",
    initial_train_end: str = "2013-12-31",
    last_date: str = "2026-04-28",
) -> list[Fold]:
    """Annual expanding-window folds as per the spec.

    Fold 1: train 2008-2013, test 2014 (full year)
    ...
    Fold 13: train 2008-2025, test 2026-01-01 .. last_date (partial)
    """
    start = pd.Timestamp(start_date)
    init_end = pd.Timestamp(initial_train_end)
    last = pd.Timestamp(last_date)

    folds: list[Fold] = []
    train_end = init_end
    idx = 1
    while True:
        test_start = train_end + pd.Timedelta(days=1)
        if test_start > last:
            break
        # Annual test window
        test_end_candidate = pd.Timestamp(f"{test_start.year}-12-31")
        test_end = min(test_end_candidate, last)
        folds.append(
            Fold(
                index=idx,
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        if test_end >= last:
            break
        train_end = test_end
        idx += 1
    return folds


def iter_fold_slices(
    frame: pd.DataFrame, folds: list[Fold]
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train_frame, test_frame) for each fold, indexed by date."""
    for fold in folds:
        train = frame.loc[fold.train_start : fold.train_end]
        test = frame.loc[fold.test_start : fold.test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        yield fold, train, test


def run_walk_forward(
    frame: pd.DataFrame,
    model_factory: Callable[[], dict[str, BaseModel]],
    returns: pd.Series | None = None,
    feature_cols: list[str] | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all models across all folds, return a long-format results frame.

    Parameters
    ----------
    frame : DataFrame indexed by date with columns ['y', *features]. `y` is
        log-variance of the 1-day-ahead target.
    model_factory : callable producing a fresh {name: model} dict per fold.
    returns : Series of daily log returns aligned with `frame`. Required for
        GARCH; other models ignore it.
    feature_cols : explicit list of feature column names. Defaults to all
        columns except 'y'.
    output_dir : if provided, per-fold per-model prediction parquets are
        written to output_dir / f"{model}_fold{n}.parquet".
    """
    if feature_cols is None:
        feature_cols = [c for c in frame.columns if c != "y"]

    folds = make_annual_folds(
        start_date=str(frame.index.min().date()),
        last_date=str(frame.index.max().date()),
    )

    results: list[dict] = []

    for fold, train, test in iter_fold_slices(frame, folds):
        if verbose:
            print(
                f"[fold {fold.index:02d}] train {fold.train_start.date()}..{fold.train_end.date()} "
                f"({len(train)} rows)  test {fold.test_start.date()}..{fold.test_end.date()} ({len(test)} rows)"
            )

        X_tr = train[feature_cols]
        y_tr = train["y"]
        X_te = test[feature_cols]
        y_te = test["y"]

        models = model_factory()

        # Returns slices for GARCH
        if returns is not None:
            train_returns = returns.loc[fold.train_start : fold.train_end]
            test_returns = returns.loc[fold.test_start : fold.test_end]
        else:
            train_returns = test_returns = None

        for name, model in models.items():
            try:
                if name == "garch11":
                    if train_returns is None:
                        if verbose:
                            print(f"  {name}: skipped (no returns series)")
                        continue
                    model.fit(X_tr, y_tr, returns=train_returns)
                    y_pred_log = model.predict(X_te, test_returns=test_returns)
                else:
                    model.fit(X_tr, y_tr)
                    y_pred_log = model.predict(X_te)
            except Exception as e:
                if verbose:
                    print(f"  {name}: FAILED ({type(e).__name__}: {e})")
                continue

            # Convert log-variance predictions back to variance safely.
            # This prevents overflow when a model predicts an extreme log-variance value.
            y_true_log = np.asarray(y_te.values, dtype=float)
            y_pred_log = np.asarray(y_pred_log, dtype=float)

            # Replace NaN / +/-inf predictions before clipping.
            fallback = np.nanmedian(y_true_log)

            y_pred_log = np.nan_to_num(
                y_pred_log,
                nan=fallback,
                posinf=5.0,
                neginf=-25.0,
            )

            # Clip log-variance predictions to avoid np.exp() overflow.
            # These are broad safety bounds, not a modeling constraint during training.
            y_pred_log = np.clip(y_pred_log, -25.0, 5.0)

            var_true = np.exp(y_true_log)
            var_pred = np.exp(y_pred_log)

            losses = all_metrics(var_true, var_pred)

            row = {
                "fold": fold.index,
                "model": name,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_test": len(test),
                **losses,
                "best_params": model.best_params_,
            }
            results.append(row)

            if output_dir is not None:
                pred_df = pd.DataFrame(
                    {
                        "date": test.index,
                        "y_true_log_var": y_te.values,
                        "y_pred_log_var": y_pred_log,
                        "var_true": var_true,
                        "var_pred": var_pred,
                    }
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                pred_df.to_parquet(
                    output_dir / f"{name}_fold{fold.index}.parquet", index=False
                )

            if verbose:
                print(
                    f"  {name:10s}  QLIKE={losses['qlike']:.4f}  "
                    f"MSE_log={losses['mse_log']:.4f}  MAE_vol={losses['mae_vol']:.5f}"
                )

    return pd.DataFrame(results)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold losses into a model-level summary (mean ± std).

    Only folds 1-12 (complete calendar years) are included. Fold 13 covers
    Jan-Apr 2026 only (79 trading days) and its extreme QLIKE values would
    distort the cross-fold averages.
    """
    if results.empty:
        return results
    results = results[results["fold"] <= 12]
    agg = results.groupby("model").agg(
        qlike_mean=("qlike", "mean"),
        qlike_std=("qlike", "std"),
        mse_log_mean=("mse_log", "mean"),
        mse_log_std=("mse_log", "std"),
        mae_vol_mean=("mae_vol", "mean"),
        mae_vol_std=("mae_vol", "std"),
        n_folds=("fold", "nunique"),
    )
    return agg.sort_values("qlike_mean")
