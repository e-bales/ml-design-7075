import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DIR = Path("data") / "raw"
DEFAULT_OUTPUT_DIR = Path("data") / "processed"
DEFAULT_MACRO_PATH = Path("data") / "raw" / "macro" / "macro_features.csv"


def load_raw_data(ticker: str, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_dir = raw_dir / ticker.upper()
    price_path = ticker_dir / "prices_daily.csv"
    news_path = ticker_dir / "news_sentiment.csv"

    if not price_path.exists():
        raise FileNotFoundError(f"Missing price file: {price_path}")
    if not news_path.exists():
        raise FileNotFoundError(f"Missing news file: {news_path}")

    price_df = pd.read_csv(price_path)

    try:
        news_df = pd.read_csv(news_path)
    except pd.errors.EmptyDataError:
        print(f"  Warning: {news_path} is empty. Proceeding with no news data.")
        news_df = pd.DataFrame()

    return price_df, news_df


def build_daily_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df = news_df.copy()

    if news_df.empty:
        columns = [
            "date",
            "article_count",
            "avg_sentiment",
            "median_sentiment",
            "sentiment_std",
            "avg_relevance",
            "weighted_sentiment",
            "bullish_share",
            "bearish_share",
            "neutral_share",
        ]
        return pd.DataFrame(columns=columns)

    news_df["time_published"] = pd.to_datetime(news_df["time_published"], errors="coerce")
    news_df["overall_sentiment_score"] = pd.to_numeric(
        news_df["overall_sentiment_score"], errors="coerce"
    )
    news_df["relevance_score"] = pd.to_numeric(news_df["relevance_score"], errors="coerce")
    news_df["ticker_sentiment_score"] = pd.to_numeric(
        news_df["ticker_sentiment_score"], errors="coerce"
    )
    news_df["date"] = news_df["time_published"].dt.floor("D")

    news_df["is_bullish"] = news_df["ticker_sentiment_label"].isin(
        ["Bullish", "Somewhat-Bullish"]
    ).astype(int)
    news_df["is_bearish"] = news_df["ticker_sentiment_label"].isin(
        ["Bearish", "Somewhat-Bearish"]
    ).astype(int)
    news_df["is_neutral"] = (news_df["ticker_sentiment_label"] == "Neutral").astype(int)
    news_df["weighted_sent_component"] = (
        news_df["ticker_sentiment_score"] * news_df["relevance_score"]
    )

    daily_sentiment = (
        news_df.groupby("date")
        .agg(
            article_count=("title", "count"),
            avg_sentiment=("ticker_sentiment_score", "mean"),
            median_sentiment=("ticker_sentiment_score", "median"),
            sentiment_std=("ticker_sentiment_score", "std"),
            avg_relevance=("relevance_score", "mean"),
            sum_relevance=("relevance_score", "sum"),
            weighted_sent_sum=("weighted_sent_component", "sum"),
            bullish_share=("is_bullish", "mean"),
            bearish_share=("is_bearish", "mean"),
            neutral_share=("is_neutral", "mean"),
        )
        .reset_index()
    )

    daily_sentiment["weighted_sentiment"] = (
        daily_sentiment["weighted_sent_sum"] / daily_sentiment["sum_relevance"]
    )
    daily_sentiment = daily_sentiment.drop(columns=["sum_relevance", "weighted_sent_sum"])
    daily_sentiment["sentiment_std"] = daily_sentiment["sentiment_std"].fillna(0)

    return daily_sentiment.sort_values("date").reset_index(drop=True)


def build_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.copy()

    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    price_df = price_df.sort_values("date").reset_index(drop=True)

    price_df["return_1d"] = price_df["close"].pct_change()
    price_df["return_3d"] = price_df["close"].pct_change(3)
    price_df["return_5d"] = price_df["close"].pct_change(5)
    price_df["volatility_5d"] = price_df["return_1d"].rolling(5).std()
    price_df["volume_change_1d"] = price_df["volume"].pct_change()
    price_df["hl_spread"] = (price_df["high"] - price_df["low"]) / price_df["close"]

    # Moving averages
    price_df["sma_5"] = price_df["close"].rolling(5).mean()
    price_df["sma_10"] = price_df["close"].rolling(10).mean()
    price_df["sma_20"] = price_df["close"].rolling(20).mean()

    # RSI (14-period)
    delta = price_df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    price_df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = price_df["close"].ewm(span=12, adjust=False).mean()
    ema26 = price_df["close"].ewm(span=26, adjust=False).mean()
    price_df["macd"] = ema12 - ema26
    price_df["macd_signal"] = price_df["macd"].ewm(span=9, adjust=False).mean()
    price_df["macd_hist"] = price_df["macd"] - price_df["macd_signal"]

    # Target: next-day return and direction (next_close is intermediate only)
    next_close = price_df["close"].shift(-1)
    price_df["next_day_return"] = (next_close - price_df["close"]) / price_df["close"]
    price_df["target_up"] = (price_df["next_day_return"] > 0).astype(int)

    return price_df


def load_macro_data(macro_path: Path) -> pd.DataFrame | None:
    if not macro_path.exists():
        return None
    macro_df = pd.read_csv(macro_path)
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    return macro_df


def build_modeling_table(
    price_df: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    model_df = price_df.merge(daily_sentiment, on="date", how="left")

    macro_cols = []
    if macro_df is not None:
        macro_cols = [c for c in macro_df.columns if c != "date"]
        model_df = model_df.merge(macro_df, on="date", how="left")
        # Forward-fill any gaps from FRED (holidays, missing observations)
        model_df[macro_cols] = model_df[macro_cols].ffill()

    # has_news=1 when sentiment data exists for that day, 0 when no news was found
    model_df["has_news"] = model_df["article_count"].notna().astype(int)

    sentiment_cols = [
        "article_count",
        "avg_sentiment",
        "median_sentiment",
        "sentiment_std",
        "avg_relevance",
        "weighted_sentiment",
        "bullish_share",
        "bearish_share",
        "neutral_share",
    ]

    for col in sentiment_cols:
        model_df[col] = model_df[col].fillna(0).infer_objects(copy=False)

    model_df = model_df.dropna(
        subset=[
            "return_1d",
            "return_3d",
            "return_5d",
            "volatility_5d",
            "volume_change_1d",
            "hl_spread",
            "sma_20",
            "rsi_14",
            "next_day_return",
        ] + macro_cols
    ).copy()

    return model_df.sort_values("date").reset_index(drop=True)


def save_processed_outputs(
    ticker: str,
    daily_sentiment: pd.DataFrame,
    model_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    ticker_dir = output_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    daily_sentiment_path = ticker_dir / "daily_sentiment_features.csv"
    modeling_table_path = ticker_dir / "modeling_table.csv"

    daily_sentiment.to_csv(daily_sentiment_path, index=False)
    model_df.to_csv(modeling_table_path, index=False)

    print()
    print("Saved processed files:")
    print(f"  {daily_sentiment_path}")
    print(f"  {modeling_table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed feature datasets from raw price and news data."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing raw pipeline outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for processed feature outputs.",
    )
    parser.add_argument(
        "--macro-path",
        default=str(DEFAULT_MACRO_PATH),
        help="Path to macro_features.csv from macro.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    macro_path = Path(args.macro_path)

    macro_df = load_macro_data(macro_path)
    if macro_df is not None:
        print(f"Loaded macro features: {len(macro_df.columns) - 1} columns.")
    else:
        print("No macro data found — run macro.py first to add macro features.")

    print(f"Loading raw data for {args.ticker}...")
    price_df, news_df = load_raw_data(args.ticker, raw_dir)
    print(f"Loaded {len(price_df)} price rows and {len(news_df)} news rows.")

    print()
    print("Building daily sentiment features...")
    daily_sentiment = build_daily_sentiment_features(news_df)
    print(f"Created {len(daily_sentiment)} daily sentiment rows.")

    print()
    print("Building price features...")
    price_features_df = build_price_features(price_df)
    print(f"Created {len(price_features_df)} price feature rows.")

    print()
    print("Merging into modeling table...")
    model_df = build_modeling_table(price_features_df, daily_sentiment, macro_df)
    print(f"Created {len(model_df)} modeling rows.")

    save_processed_outputs(args.ticker, daily_sentiment, model_df, output_dir)


if __name__ == "__main__":
    main()
