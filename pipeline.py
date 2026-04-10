import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
DEFAULT_ENV_PATH = Path(".env")
DEFAULT_OUTPUT_DIR = Path("data") / "raw"
DEFAULT_ARCHIVE_DIR = Path("past_data_pulled")


@dataclass
class ChunkResult:
    start: pd.Timestamp
    end: pd.Timestamp
    row_count: int
    is_truncated: bool
    window_days: float


class AlphaVantageDailyLimitError(RuntimeError):
    pass


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


def get_api_key() -> str:
    accepted_names = [
        "ALPHA_VANTAGE_API_KEY",
        "ALPHAVANTAGE_API_KEY",
        "ALPHA_VANTAGE_KEY",
        "ALPHAVANTAGE_KEY",
        "API_KEY",
    ]

    api_key = None
    for env_name in accepted_names:
        if os.getenv(env_name):
            api_key = os.getenv(env_name)
            break

    if not api_key:
        raise ValueError(
            "Missing Alpha Vantage API key. Set one of these in .env or your environment: "
            + ", ".join(accepted_names)
        )
    return api_key


def alpha_vantage_get(params: dict) -> dict:
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(data["Error Message"])

        info_message = data.get("Information", "")
        if info_message and (
            "rate limit" in info_message.lower()
            or "1 request per second" in info_message.lower()
            or "25 requests per day" in info_message.lower()
        ):
            if "25 requests per day" in info_message.lower():
                raise AlphaVantageDailyLimitError(info_message)

            if attempt == max_attempts:
                raise RuntimeError(info_message)

            wait_seconds = max(2.0, attempt * 2.0)
            print(
                f"Alpha Vantage rate limit hit. Waiting {wait_seconds:.1f}s before retry "
                f"({attempt}/{max_attempts})..."
            )
            time.sleep(wait_seconds)
            continue

        return data

    raise RuntimeError("Alpha Vantage request failed after retries.")


def fetch_daily_prices(symbol: str, api_key: str, outputsize: str) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    data = alpha_vantage_get(params)
    ts_key = "Time Series (Daily)"

    if ts_key not in data:
        raise ValueError(f"Unexpected price response for {symbol}: {data}")

    price_df = (
        pd.DataFrame(data[ts_key]).T.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    price_df["date"] = pd.to_datetime(price_df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    return price_df.sort_values("date").reset_index(drop=True)


def fetch_news_chunk(
    symbol: str,
    api_key: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    limit: int,
) -> tuple[pd.DataFrame, ChunkResult]:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "time_from": start_dt.strftime("%Y%m%dT%H%M"),
        "time_to": end_dt.strftime("%Y%m%dT%H%M"),
        "limit": limit,
        "sort": "LATEST",
        "apikey": api_key,
    }

    data = alpha_vantage_get(params)
    feed = data.get("feed", [])
    rows = []

    for article in feed:
        row = {
            "time_published": article.get("time_published"),
            "title": article.get("title"),
            "url": article.get("url"),
            "source": article.get("source"),
            "summary": article.get("summary"),
            "overall_sentiment_score": article.get("overall_sentiment_score"),
            "overall_sentiment_label": article.get("overall_sentiment_label"),
            "relevance_score": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None,
        }

        for ticker_entry in article.get("ticker_sentiment", []):
            if ticker_entry.get("ticker") == symbol:
                row["relevance_score"] = ticker_entry.get("relevance_score")
                row["ticker_sentiment_score"] = ticker_entry.get("ticker_sentiment_score")
                row["ticker_sentiment_label"] = ticker_entry.get("ticker_sentiment_label")
                break

        rows.append(row)

    chunk_df = pd.DataFrame(rows)
    if not chunk_df.empty:
        chunk_df["time_published"] = pd.to_datetime(
            chunk_df["time_published"],
            format="%Y%m%dT%H%M%S",
            errors="coerce",
        )
        chunk_df["overall_sentiment_score"] = pd.to_numeric(
            chunk_df["overall_sentiment_score"],
            errors="coerce",
        )
        chunk_df["relevance_score"] = pd.to_numeric(
            chunk_df["relevance_score"],
            errors="coerce",
        )
        chunk_df["ticker_sentiment_score"] = pd.to_numeric(
            chunk_df["ticker_sentiment_score"],
            errors="coerce",
        )
        chunk_df = chunk_df.sort_values("time_published").reset_index(drop=True)

    return chunk_df, ChunkResult(
        start=start_dt,
        end=end_dt,
        row_count=len(chunk_df),
        is_truncated=len(chunk_df) >= limit,
        window_days=(end_dt - start_dt).total_seconds() / 86400,
    )


def fetch_news_history(
    symbol: str,
    api_key: str,
    months_back: int,
    chunk_days: int,
    limit: int,
    pause_seconds: float,
    max_split_depth: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, str | None]:
    end_dt = pd.Timestamp.now().floor("min")
    start_boundary = end_dt - pd.DateOffset(months=months_back)

    all_chunks = []
    chunk_summaries = []
    window_counter = 0
    daily_limit_hit = False
    daily_limit_message = None

    def collect_window(start_dt: pd.Timestamp, end_dt: pd.Timestamp, depth: int = 0) -> None:
        nonlocal window_counter
        window_counter += 1
        window_id = window_counter

        try:
            chunk_df, summary = fetch_news_chunk(symbol, api_key, start_dt, end_dt, limit)
        except AlphaVantageDailyLimitError as exc:
            raise AlphaVantageDailyLimitError(
                f"{exc} Last attempted window: {start_dt} -> {end_dt}"
            ) from exc

        indent = "  " * depth
        print(
            f"{indent}Window {window_id}: "
            f"{start_dt.strftime('%Y-%m-%d %H:%M')} -> {end_dt.strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"{indent}  Raw articles returned: {summary.row_count}")
        if summary.row_count:
            print(f"{indent}  Oldest: {chunk_df['time_published'].min()}")
            print(f"{indent}  Newest: {chunk_df['time_published'].max()}")

        should_split = summary.is_truncated and summary.window_days > 7 and depth < max_split_depth
        chunk_summaries.append(
            {
                "window_number": window_id,
                "window_start": summary.start,
                "window_end": summary.end,
                "window_days": summary.window_days,
                "row_count": summary.row_count,
                "is_truncated": summary.is_truncated,
                "was_split": should_split,
                "split_depth": depth,
            }
        )

        if should_split:
            midpoint = start_dt + (end_dt - start_dt) / 2
            midpoint = midpoint.floor("min")

            if midpoint <= start_dt or midpoint >= end_dt:
                print(f"{indent}  Warning: unable to split this window further.")
                all_chunks.append(chunk_df)
                return

            print(
                f"{indent}  Window hit the row limit. Splitting into smaller windows "
                f"for better coverage."
            )
            time.sleep(pause_seconds)
            collect_window(start_dt, midpoint, depth + 1)
            time.sleep(pause_seconds)
            collect_window(midpoint + pd.Timedelta(minutes=1), end_dt, depth + 1)
            return

        if summary.is_truncated:
            print(
                f"{indent}  Warning: window hit the row limit at the minimum split size "
                f"and may still be truncated."
            )

        all_chunks.append(chunk_df)

    current_end = end_dt
    try:
        while current_end > start_boundary:
            current_start = max(start_boundary, current_end - pd.Timedelta(days=chunk_days))
            collect_window(current_start, current_end)
            current_end = current_start - pd.Timedelta(minutes=1)
            time.sleep(pause_seconds)
    except AlphaVantageDailyLimitError as exc:
        daily_limit_hit = True
        daily_limit_message = str(exc)
        print()
        print("Daily Alpha Vantage request limit reached.")
        print("Saving the data collected so far.")

    news_df = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    if not news_df.empty:
        news_df = news_df.drop_duplicates(subset=["url"]).copy()
        news_df = news_df.drop_duplicates(
            subset=["time_published", "title", "source"]
        ).copy()
        news_df = news_df.sort_values("time_published").reset_index(drop=True)

    chunk_summary_df = pd.DataFrame(chunk_summaries)
    if not chunk_summary_df.empty:
        chunk_summary_df["daily_limit_hit"] = daily_limit_hit
        chunk_summary_df["daily_limit_message"] = daily_limit_message

    return news_df, chunk_summary_df, daily_limit_hit, daily_limit_message


def write_outputs(
    symbol: str,
    price_df: pd.DataFrame,
    news_df: pd.DataFrame,
    chunk_summary_df: pd.DataFrame,
    output_dir: Path,
    archive_dir: Path,
) -> None:
    symbol_dir = output_dir / symbol.upper()
    symbol_dir.mkdir(parents=True, exist_ok=True)

    price_path = symbol_dir / "prices_daily.csv"
    news_path = symbol_dir / "news_sentiment.csv"
    summary_path = symbol_dir / "news_chunk_summary.csv"

    price_df.to_csv(price_path, index=False)
    news_df.to_csv(news_path, index=False)
    chunk_summary_df.to_csv(summary_path, index=False)

    archive_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    archive_prefix = f"{symbol.upper()}_{run_stamp}"

    archive_price_path = archive_dir / f"{archive_prefix}_prices_daily.csv"
    archive_news_path = archive_dir / f"{archive_prefix}_news_sentiment.csv"
    archive_summary_path = archive_dir / f"{archive_prefix}_news_chunk_summary.csv"

    price_df.to_csv(archive_price_path, index=False)
    news_df.to_csv(archive_news_path, index=False)
    chunk_summary_df.to_csv(archive_summary_path, index=False)

    print()
    print("Saved files:")
    print(f"  {price_path}")
    print(f"  {news_path}")
    print(f"  {summary_path}")
    print()
    print("Archived copies:")
    print(f"  {archive_price_path}")
    print(f"  {archive_news_path}")
    print(f"  {archive_summary_path}")


def fetch_macro_series(function: str, api_key: str, interval: str | None = None) -> pd.DataFrame:
    params = {"function": function, "apikey": api_key}
    if interval:
        params["interval"] = interval
    data = alpha_vantage_get(params)

    rows = data.get("data", [])
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df


def write_macro_outputs(
    fed_funds_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    macro_dir = output_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)

    fed_path = macro_dir / "federal_funds_rate.csv"
    cpi_path = macro_dir / "cpi.csv"

    fed_funds_df.to_csv(fed_path, index=False)
    cpi_df.to_csv(cpi_path, index=False)

    print()
    print("Saved macro files:")
    print(f"  {fed_path}")
    print(f"  {cpi_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Alpha Vantage daily prices and news sentiment."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--price-outputsize",
        default="compact",
        choices=["compact", "full"],
        help="Price history size. Free Alpha Vantage keys typically support compact.",
    )
    parser.add_argument(
        "--news-months",
        type=int,
        default=6,
        help="Months of news history to collect. Defaults to 6.",
    )
    parser.add_argument(
        "--news-chunk-days",
        type=int,
        default=30,
        help="Chunk size in days for news collection.",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=1000,
        help="Max news rows per Alpha Vantage request.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.5,
        help="Pause between news requests to be gentle with rate limits.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for raw CSV outputs.",
    )
    parser.add_argument(
        "--archive-dir",
        default=str(DEFAULT_ARCHIVE_DIR),
        help="Directory for timestamped archived CSV pulls.",
    )
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Skip fetching macro data (fed funds rate, CPI). Use on 2nd+ ticker runs in the same day to save API requests.",
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip fetching price data and reuse existing prices_daily.csv. Saves 1 API request per ticker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(DEFAULT_ENV_PATH)
    api_key = get_api_key()
    output_dir = Path(args.output_dir)
    archive_dir = Path(args.archive_dir)

    if args.skip_prices:
        price_path = output_dir / args.ticker.upper() / "prices_daily.csv"
        if not price_path.exists():
            raise FileNotFoundError(f"--skip-prices set but no existing price file found at {price_path}")
        price_df = pd.read_csv(price_path)
        price_df["date"] = pd.to_datetime(price_df["date"])
        print(f"Loaded existing prices for {args.ticker} ({len(price_df)} rows).")
    else:
        print(f"Fetching daily prices for {args.ticker}...")
        price_df = fetch_daily_prices(args.ticker, api_key, args.price_outputsize)
        print(f"Retrieved {len(price_df)} daily price rows.")

    print()
    print(f"Fetching news sentiment for {args.ticker}...")
    news_df, chunk_summary_df, daily_limit_hit, daily_limit_message = fetch_news_history(
        symbol=args.ticker,
        api_key=api_key,
        months_back=args.news_months,
        chunk_days=args.news_chunk_days,
        limit=args.news_limit,
        pause_seconds=args.pause_seconds,
    )
    print()
    print(f"Retrieved {len(news_df)} deduplicated news rows.")

    if daily_limit_hit:
        print("News ingestion stopped early because the free daily request limit was reached.")
        print(daily_limit_message)

    write_outputs(
        args.ticker,
        price_df,
        news_df,
        chunk_summary_df,
        output_dir,
        archive_dir,
    )

    if not args.skip_macro:
        print()
        print("Fetching macro data (fed funds rate + CPI)...")
        try:
            fed_funds_df = fetch_macro_series("FEDERAL_FUNDS_RATE", api_key, interval="daily")
            cpi_df = fetch_macro_series("CPI", api_key, interval="monthly")
            write_macro_outputs(fed_funds_df, cpi_df, output_dir)
        except AlphaVantageDailyLimitError:
            print("Daily API limit reached while fetching macro data.")
            print("Ticker data was already saved. Re-run with --skip-macro tomorrow to get macro data.")


if __name__ == "__main__":
    main()
