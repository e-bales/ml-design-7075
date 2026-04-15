from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

FEATURE_COLUMNS = [
    "prev_close",
    "daily_return",
    "ma_3",
    "ma_diff",
    "sentiment_compound",
    "sentiment_pos",
    "sentiment_neu",
    "sentiment_neg",
]


def build_features(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    else:
        raise ValueError("Raw data must include a 'date' column.")

    df["prev_close"] = df["close"].shift(1)
    df["daily_return"] = df["close"].pct_change()
    df["ma_3"] = df["close"].rolling(window=3, min_periods=1).mean()
    df["ma_diff"] = df["close"] - df["ma_3"]
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    if ticker is not None:
        df["ticker"] = ticker.upper()
    elif "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"

    sentiment_columns = [
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neu",
        "sentiment_neg",
    ]
    for col in sentiment_columns:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=["prev_close", "daily_return", "target"]).reset_index(drop=True)
    return df


def save_processed_features(
    raw_file: Path,
    processed_dir: Path,
    ticker: Optional[str] = None,
) -> Path:
    processed_dir = processed_dir.resolve()
    versions_dir = processed_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_file)
    ticker_value = ticker or raw_file.stem.split("_")[0]
    processed_df = build_features(df, ticker=ticker_value)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = versions_dir / f"{ticker_value}_processed_{timestamp}.csv"
    processed_df.to_csv(output_file, index=False)
    return output_file
