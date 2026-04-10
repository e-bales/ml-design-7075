import argparse
import os
from pathlib import Path

import pandas as pd
from fredapi import Fred


DEFAULT_OUTPUT_DIR = Path("data") / "raw" / "macro"
DEFAULT_ENV_PATH = Path(".env")

# FRED series to pull: {output_column_name: FRED_series_id}
FRED_SERIES = {
    "vix":               "VIXCLS",
    "fed_funds_rate":    "DFF",
    "yield_curve":       "T10Y2Y",
    "cpi":               "CPIAUCSL",
    "unemployment":      "UNRATE",
    "sp500":             "SP500",
    "oil_wti":           "DCOILWTICO",
    "high_yield_spread": "BAMLH0A0HYM2",
}


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_fred_api_key() -> str:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError(
            "Missing FRED API key. Set FRED_API_KEY in your .env file."
        )
    return key


def pull_series(fred: Fred, name: str, series_id: str, start: str) -> pd.Series:
    print(f"  Pulling {name} ({series_id})...")
    raw = fred.get_series(series_id, observation_start=start)
    raw.name = name
    return raw


def build_macro_features(fred: Fred, start: str) -> pd.DataFrame:
    series = {}
    for name, series_id in FRED_SERIES.items():
        s = pull_series(fred, name, series_id, start)
        series[name] = s

    # Combine into daily DataFrame aligned to business days
    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Forward-fill gaps (weekends, holidays, monthly series carried forward)
    df = df.asfreq("B").ffill()

    # SP500 daily return is more useful than the index level
    df["sp500_return_1d"] = df["sp500"].pct_change()
    df = df.drop(columns=["sp500"])

    # VIX 5-day rolling average smooths day-to-day noise
    df["vix_ma5"] = df["vix"].rolling(5).mean()

    df = df.reset_index()
    return df.dropna(subset=["vix", "fed_funds_rate"]).reset_index(drop=True)


def save_outputs(macro_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "macro_features.csv"
    macro_df.to_csv(path, index=False)
    print()
    print(f"Saved macro features: {path}")
    print(f"  {len(macro_df)} rows | {len(macro_df.columns) - 1} features")
    print(f"  Date range: {macro_df['date'].min().date()} to {macro_df['date'].max().date()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull macro features from FRED and save to CSV."
    )
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date for FRED series pull (YYYY-MM-DD). Default 2024-01-01.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save macro_features.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(DEFAULT_ENV_PATH)
    api_key = get_fred_api_key()
    fred = Fred(api_key=api_key)

    print(f"Pulling {len(FRED_SERIES)} FRED series from {args.start}...")
    macro_df = build_macro_features(fred, start=args.start)
    save_outputs(macro_df, Path(args.output_dir))


if __name__ == "__main__":
    main()
