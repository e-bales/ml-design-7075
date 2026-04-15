from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_api_key() -> str:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        if load_dotenv is not None:
            load_dotenv(env_path)
        else:
            _load_env_file(env_path)

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Alpha Vantage API key is missing. Set the ALPHAVANTAGE_API_KEY environment variable or add it to .env."
        )
    return api_key


def _parse_time_series(json_payload: dict) -> pd.DataFrame:
    if "Error Message" in json_payload:
        raise ValueError(json_payload["Error Message"])
    if "Note" in json_payload:
        raise ValueError(json_payload["Note"])
    if "Information" in json_payload:
        raise ValueError(json_payload["Information"])

    time_series_key = next(
        (key for key in json_payload.keys() if "Time Series" in key),
        None,
    )
    time_series = json_payload.get(time_series_key) if time_series_key else None
    if time_series is None:
        raise ValueError(
            "Unexpected Alpha Vantage response format. "
            f"Response keys: {list(json_payload.keys())}. "
            "Check your API key, request rate limit, or symbol name."
        )

    rows = []
    for date_str, daily_data in time_series.items():
        rows.append(
            {
                "date": pd.to_datetime(date_str),
                "open": float(daily_data["1. open"]),
                "high": float(daily_data["2. high"]),
                "low": float(daily_data["3. low"]),
                "close": float(daily_data["4. close"]),
                "adjusted_close": float(daily_data.get("5. adjusted close", daily_data["4. close"])),
                "volume": int(daily_data["5. volume"] if "5. volume" in daily_data else daily_data["6. volume"]),
                "dividend_amount": float(daily_data.get("7. dividend amount", 0.0)),
                "split_coefficient": float(daily_data.get("8. split coefficient", 1.0)),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_daily_adjusted(symbol: str, outputsize: str = "compact", api_key: str | None = None) -> pd.DataFrame:
    """Fetch daily stock price data from Alpha Vantage using the free daily endpoint."""
    api_key = api_key or get_api_key()
    function_name = "TIME_SERIES_DAILY"
    params = {
        "function": function_name,
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "json",
        "apikey": api_key,
    }

    for attempt in range(1, 7):
        response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        try:
            return _parse_time_series(payload)
        except ValueError as exc:
            message = str(exc).lower()
            if "premium endpoint" in message or "premium" in message:
                # Already on the free endpoint, so no premium fallback exists.
                raise

            if "1 request per second" in message or "request rate" in message or "rate limit" in message:
                if attempt == 6:
                    raise
                wait_seconds = min(10, attempt * 2)
                time.sleep(wait_seconds)
                continue

            raise

    raise RuntimeError("Alpha Vantage request failed after multiple retries.")


def save_daily_adjusted(symbol: str, raw_dir: Path, outputsize: str = "compact") -> Path:
    raw_dir = raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_daily_adjusted(symbol, outputsize=outputsize)
    output_file = raw_dir / f"{symbol.upper()}_daily.csv"
    df.to_csv(output_file, index=False)
    return output_file
