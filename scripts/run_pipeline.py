"""End-to-end pipeline:

  1. JSON -> raw CSV (if not already converted)
  2. Build feature matrix + target -> data/processed/spy_features.parquet
  3. Walk-forward validation for all 6 models across 13 annual folds
  4. Write per-fold predictions to results/predictions/
  5. Write summary table to results/results_summary.csv

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import SEED
from src.features import build_feature_matrix, prepare_supervised_frame
from src.models import build_models
from src.validation import run_walk_forward, summarize_results


RAW_CSV = ROOT / "data" / "raw" / "spy_ohlc.csv"
FEATURES_PARQUET = ROOT / "data" / "processed" / "spy_features.parquet"
PREDICTIONS_DIR = ROOT / "results" / "predictions"
SUMMARY_CSV = ROOT / "results" / "results_summary.csv"
PER_FOLD_CSV = ROOT / "results" / "per_fold_losses.csv"


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pass


def _ensure_raw_csv() -> pd.DataFrame:
    if not RAW_CSV.exists():
        print(f"Raw CSV not found at {RAW_CSV} — running converter...")
        from scripts.convert_json_to_csv import main as convert

        convert()
    df = pd.read_csv(RAW_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _build_features(ohlc: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("Building feature matrix...")
    X, y = build_feature_matrix(ohlc)

    # Returns series aligned with features, for GARCH
    close = ohlc.set_index("date")["close"].sort_index()
    log_returns = np.log(close / close.shift(1))

    frame = X.copy()
    frame.insert(0, "y", y)
    frame = frame.dropna()

    # Save the processed frame
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().rename(columns={"index": "date"}).to_parquet(
        FEATURES_PARQUET, index=False
    )
    print(
        f"Features written to {FEATURES_PARQUET}  "
        f"shape={frame.shape}  date_range={frame.index.min().date()}..{frame.index.max().date()}"
    )
    return frame, log_returns.loc[frame.index]


def main() -> None:
    _seed_everything()

    ohlc = _ensure_raw_csv()
    frame, returns = _build_features(ohlc)

    print("\nStarting walk-forward validation...")
    results = run_walk_forward(
        frame=frame,
        model_factory=build_models,
        returns=returns,
        output_dir=PREDICTIONS_DIR,
        verbose=True,
    )

    if results.empty:
        print("No results produced — aborting.")
        return

    # Persist per-fold details and summary
    PER_FOLD_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.drop(columns=["best_params"]).to_csv(PER_FOLD_CSV, index=False)
    summary = summarize_results(results)
    summary.to_csv(SUMMARY_CSV)

    print("\n=== Master results (sorted by QLIKE) ===")
    print(summary.to_string())
    print(f"\nWrote {PER_FOLD_CSV}")
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
