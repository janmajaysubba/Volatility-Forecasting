"""Convert the raw Godel Terminal JSON export to spy_ohlc.csv.

Usage:
    python scripts/convert_json_to_csv.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "Historical_Prices_ETC_SPY_2026-04-28.json"
CSV_PATH = ROOT / "data" / "raw" / "spy_ohlc.csv"


def main() -> None:
    with JSON_PATH.open("r") as f:
        records = json.load(f)

    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["UTCDate"], utc=True).dt.tz_convert(None).dt.normalize()
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date", keep="last")

    # Sanity
    assert df["date"].is_monotonic_increasing
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df["low"]).all()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {CSV_PATH}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    main()
