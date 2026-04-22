"""
api.py — FastAPI backend serving next-day stock direction predictions.

Startup: trains per-ticker models using historical processed data.
/predict/{ticker}: fetches live Alpha Vantage data, engineers features, returns prediction.

Run:
    uvicorn api:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from analyze import compute_ticker_metrics
from features import (
    build_daily_sentiment_features,
    build_price_features,
    load_macro_data,
)
from model import (
    DEFAULT_PROCESSED_DIR,
    MACRO_FEATURE_COLS,
    PRICE_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
    TARGET_COL,
    USE_MACRO,
    build_models,
    get_feature_cols_single_ticker,
    load_all_tickers,
    prepare_features,
    time_split,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "AMD", "BA", "DIS", "JPM", "NFLX", "PFE", "V", "WMT"]
MACRO_PATH = Path("data/raw/macro/macro_features.csv")
RAW_DIR = Path("data/raw")

# Populated dynamically at startup by compute_ticker_metrics()
BEST_MODEL: dict[str, str] = {}
HISTORICAL_PERF: dict[str, dict] = {}

# Trained model store: ticker -> model_name -> (fitted_model, scaler_or_None, feature_cols)
_trained: dict = {}
_feature_cols: list[str] = []


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _get_api_key() -> str:
    _load_env()
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not set — add it to .env")
    return key


# ---------------------------------------------------------------------------
# Live data fetching
# ---------------------------------------------------------------------------

def _fetch_live_prices(ticker: str, api_key: str) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": api_key,
        "datatype": "json",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Time Series (Daily)" not in data:
        info = data.get("Information", data.get("Note", str(data)))
        raise HTTPException(status_code=502, detail=f"Alpha Vantage error for {ticker}: {info}")

    rows = []
    for date_str, vals in data["Time Series (Daily)"].items():
        raw_close = float(vals["4. close"])
        adj_close = float(vals["5. adjusted close"])
        ratio = adj_close / raw_close if raw_close != 0 else 1.0
        rows.append({
            "date": pd.Timestamp(date_str),
            "open": float(vals["1. open"]) * ratio,
            "high": float(vals["2. high"]) * ratio,
            "low": float(vals["3. low"]) * ratio,
            "close": adj_close,
            "volume": float(vals["6. volume"]),
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def _fetch_live_news(ticker: str, api_key: str, days: int = 30) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    time_from = (datetime.utcnow() - timedelta(days=days)).strftime("%Y%m%dT%H%M")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "limit": 200,
        "apikey": api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    feed = data.get("feed", [])
    if not feed:
        return pd.DataFrame()

    rows = []
    for article in feed:
        for ts_info in article.get("ticker_sentiment", []):
            if ts_info.get("ticker") == ticker:
                rows.append({
                    "time_published": article.get("time_published", ""),
                    "title": article.get("title", ""),
                    "overall_sentiment_score": article.get("overall_sentiment_score", 0),
                    "relevance_score": float(ts_info.get("relevance_score", 0)),
                    "ticker_sentiment_score": float(ts_info.get("ticker_sentiment_score", 0)),
                    "ticker_sentiment_label": ts_info.get("ticker_sentiment_label", "Neutral"),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Live feature engineering
# ---------------------------------------------------------------------------

def _build_live_features(ticker: str, api_key: str) -> pd.Series:
    price_df = _fetch_live_prices(ticker, api_key)
    news_df = _fetch_live_news(ticker, api_key, days=30)

    # Build price features (needs enough rows for sma_20, rsi_14, macd)
    price_feat = build_price_features(price_df)

    # Compute price-to-MA ratios (same as model.prepare_features)
    price_feat["price_to_sma5"] = price_feat["close"] / price_feat["sma_5"]
    price_feat["price_to_sma10"] = price_feat["close"] / price_feat["sma_10"]
    price_feat["price_to_sma20"] = price_feat["close"] / price_feat["sma_20"]

    # Build sentiment features and merge
    sentiment = build_daily_sentiment_features(news_df)
    merged = price_feat.merge(sentiment, on="date", how="left")

    # Fill missing sentiment with zeros
    sentiment_fill = [
        "article_count", "avg_sentiment", "median_sentiment", "sentiment_std",
        "avg_relevance", "weighted_sentiment", "bullish_share", "bearish_share", "neutral_share",
    ]
    for col in sentiment_fill:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    merged["has_news"] = merged["article_count"].notna().astype(int)

    # Macro features — use most recent local row
    macro_df = load_macro_data(MACRO_PATH)
    if macro_df is not None and USE_MACRO:
        latest_macro = macro_df.sort_values("date").iloc[-1]
        for col in MACRO_FEATURE_COLS:
            if col in latest_macro.index:
                merged[col] = latest_macro[col]
            else:
                merged[col] = 0.0

    # Drop rows that don't have all required price features (NaN from rolling windows)
    required = ["return_1d", "return_3d", "return_5d", "volatility_5d",
                "rsi_14", "price_to_sma20"]
    merged = merged.dropna(subset=required)

    if merged.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough price history to build features for {ticker}."
        )

    return merged.sort_values("date").iloc[-1]


# ---------------------------------------------------------------------------
# Startup: train per-ticker models
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _feature_cols, BEST_MODEL, HISTORICAL_PERF
    print("Loading historical data and training per-ticker models...")
    df = load_all_tickers(DEFAULT_PROCESSED_DIR)
    df = prepare_features(df)

    _feature_cols = get_feature_cols_single_ticker(df)
    train_df, _, _ = time_split(df, test_frac=0.3)

    for ticker in TICKERS:
        t_train = train_df[train_df["ticker"] == ticker]
        if t_train.empty:
            print(f"  Warning: no training data for {ticker}, skipping.")
            continue

        X_tr = t_train[_feature_cols].values
        y_tr = t_train[TARGET_COL].values
        _trained[ticker] = {}

        for run_name, model, scale in build_models():
            if scale:
                scaler = StandardScaler()
                X_fit = scaler.fit_transform(X_tr)
            else:
                scaler = None
                X_fit = X_tr
            model.fit(X_fit, y_tr)
            _trained[ticker][run_name] = (model, scaler)

        print(f"  {ticker}: trained {len(build_models())} models")

    print("Computing performance metrics from backtest...")
    BEST_MODEL, HISTORICAL_PERF = compute_ticker_metrics()
    print("All models ready.\n")
    yield


# ---------------------------------------------------------------------------
# App and routes
# ---------------------------------------------------------------------------

app = FastAPI(title="Stock Direction Prediction API", version="1.0", lifespan=lifespan)


class PredictResponse(BaseModel):
    ticker: str
    as_of_date: str
    prediction: str           # "UP" or "DOWN"
    confidence: float         # probability of UP
    model_used: str
    key_features: dict
    recent_prices: list[dict]
    historical_performance: dict


@app.get("/")
def health():
    return {"status": "ok", "models_loaded": len(_trained)}


@app.get("/tickers")
def list_tickers():
    return {
        "tickers": [
            {
                "ticker": t,
                "best_model": BEST_MODEL[t],
                "historical_performance": HISTORICAL_PERF[t],
            }
            for t in TICKERS
        ]
    }


@app.get("/predict/{ticker}", response_model=PredictResponse)
def predict(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"{ticker} not in supported tickers: {TICKERS}")
    if ticker not in _trained:
        raise HTTPException(status_code=503, detail=f"Model for {ticker} not loaded yet.")

    api_key = _get_api_key()

    # Build live features
    live_row = _build_live_features(ticker, api_key)
    as_of_date = str(live_row["date"].date()) if hasattr(live_row["date"], "date") else str(live_row["date"])

    # Assemble feature vector (handle any missing feature columns)
    feat_vector = []
    for col in _feature_cols:
        feat_vector.append(float(live_row[col]) if col in live_row.index and not pd.isna(live_row[col]) else 0.0)
    X_live = np.array(feat_vector).reshape(1, -1)

    # Predict with best model
    best_name = BEST_MODEL[ticker]
    model, scaler = _trained[ticker][best_name]
    X_input = scaler.transform(X_live) if scaler is not None else X_live
    pred_label = int(model.predict(X_input)[0])
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X_input)[0][1])
    else:
        confidence = float(pred_label)

    # Key features for display
    key_features = {
        "return_1d": round(float(live_row.get("return_1d", 0)), 4),
        "rsi_14": round(float(live_row.get("rsi_14", 50)), 2),
        "macd_hist": round(float(live_row.get("macd_hist", 0)), 4),
        "avg_sentiment": round(float(live_row.get("avg_sentiment", 0)), 4),
        "bullish_share": round(float(live_row.get("bullish_share", 0)), 4),
        "volatility_5d": round(float(live_row.get("volatility_5d", 0)), 4),
        "has_news": int(live_row.get("has_news", 0)),
    }
    if USE_MACRO:
        key_features["vix"] = round(float(live_row.get("vix", 0)), 2)
        key_features["yield_curve"] = round(float(live_row.get("yield_curve", 0)), 4)

    # Recent prices from local CSV (no extra API call)
    recent_prices = []
    price_path = RAW_DIR / ticker / "prices_daily.csv"
    if price_path.exists():
        try:
            prices = pd.read_csv(price_path, parse_dates=["date"])
            prices = prices.sort_values("date").tail(30)
            recent_prices = [
                {"date": str(row["date"].date()), "close": round(float(row["close"]), 2)}
                for _, row in prices.iterrows()
            ]
        except Exception:
            pass

    return PredictResponse(
        ticker=ticker,
        as_of_date=as_of_date,
        prediction="UP" if pred_label == 1 else "DOWN",
        confidence=round(confidence, 4),
        model_used=best_name,
        key_features=key_features,
        recent_prices=recent_prices,
        historical_performance=HISTORICAL_PERF[ticker],
    )
