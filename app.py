from pathlib import Path

import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "mlruns" / "1" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

PRICE_FEATURE_COLS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volume_change_1d",
    "hl_spread",
    "price_to_sma5",
    "price_to_sma10",
    "price_to_sma20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
]

SENTIMENT_FEATURE_COLS = [
    "has_news",
    "article_count",
    "avg_sentiment",
    "median_sentiment",
    "sentiment_std",
    "avg_relevance",
    "bullish_share",
    "bearish_share",
    "neutral_share",
    "weighted_sentiment",
]

MACRO_FEATURE_COLS = [
    "vix",
    "vix_ma5",
    "sp500_return_1d",
    "yield_curve",
    "high_yield_spread",
]
USE_MACRO = True

def find_latest_model_artifact(models_dir: Path) -> Path:
    model_dirs = [p for p in models_dir.iterdir() if p.is_dir()]
    if not model_dirs:
        raise FileNotFoundError(f"No model artifacts found in {models_dir}")
    latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
    artifact_path = latest / "artifacts"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact folder not found: {artifact_path}")
    return artifact_path


def load_latest_model(models_dir: Path):
    artifact_dir = find_latest_model_artifact(models_dir)
    model = mlflow.pyfunc.load_model(str(artifact_dir))
    return model, artifact_dir


def load_processed_data(processed_dir: Path) -> pd.DataFrame:
    dfs = []
    for ticker_dir in sorted(processed_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        modeling_table = ticker_dir / "modeling_table.csv"
        if not modeling_table.exists():
            continue
        data = pd.read_csv(modeling_table, parse_dates=["date"])
        data["ticker"] = ticker_dir.name
        dfs.append(data)

    if not dfs:
        raise FileNotFoundError(
            f"No processed modeling tables found under {processed_dir}"
        )
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    return combined


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_to_sma5"] = df["close"] / df["sma_5"]
    df["price_to_sma10"] = df["close"] / df["sma_10"]
    df["price_to_sma20"] = df["close"] / df["sma_20"]

    ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    df = pd.concat([df, ticker_dummies], axis=1)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    ticker_cols = [c for c in df.columns if c.startswith("ticker_")]
    macro_cols = [c for c in MACRO_FEATURE_COLS if c in df.columns] if USE_MACRO else []
    return PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS + macro_cols + ticker_cols

def _build_feature_frame(feature_input: dict[str, float], feature_cols: list[str], ticker_cols: list[str], ticker_col_name: str) -> pd.DataFrame:
    all_inputs = feature_input.copy()
    for tc in ticker_cols:
        all_inputs[tc] = 1.0 if tc == ticker_col_name else 0.0
    return pd.DataFrame([all_inputs], columns=feature_cols)

def get_available_tickers(processed_df: pd.DataFrame) -> list[str]:
    # print(sorted(processed_df["ticker"].dropna().unique().tolist()))
    return sorted(processed_df["ticker"].dropna().unique().tolist())
# ----------------------------
# Your existing model-loading logic
# ----------------------------
# def find_latest_model_artifact(models_dir: Path) -> Path:
#     model_dirs = [p for p in models_dir.iterdir() if p.is_dir()]
#     if not model_dirs:
#         raise FileNotFoundError(f"No model artifacts found in {models_dir}")
#     latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
#     artifact_path = latest / "artifacts"
#     if not artifact_path.exists():
#         raise FileNotFoundError(f"Model artifact folder not found: {artifact_path}")
#     return artifact_path


# def load_latest_model(models_dir: Path):
#     artifact_dir = find_latest_model_artifact(models_dir)
#     model = mlflow.pyfunc.load_model(str(artifact_dir))
#     return model, artifact_dir

def load_backend_resources():
    """
    Load the MLflow model, processed feature data, and the final feature column list.
    This should usually be called once at app startup.
    """
    model, model_path = load_latest_model(MODELS_DIR)

    raw_data = load_processed_data(PROCESSED_DIR)
    processed_df = prepare_features(raw_data)
    feature_cols = get_feature_cols(processed_df)

    return {
        "model": model,
        "model_path": model_path,
        "processed_df": processed_df,
        "feature_cols": feature_cols,
    }


def get_latest_ticker_row(ticker: str, processed_df: pd.DataFrame) -> pd.Series:
    """
    Return the most recent processed row for a given ticker.
    """
    ticker_df = processed_df[processed_df["ticker"] == ticker].copy()

    if ticker_df.empty:
        raise ValueError(f"No processed data found for ticker '{ticker}'")

    ticker_df = ticker_df.sort_values("date", ascending=False)
    return ticker_df.iloc[0]


def predict_for_ticker(
    ticker: str,
    model,
    model_path,
    processed_df: pd.DataFrame,
    feature_cols: list[str],
):
    """
    Fetch the latest row for a ticker, build the model input frame,
    run prediction, and return a JSON-friendly result.
    """
    row = get_latest_ticker_row(ticker, processed_df)
    print(row)
    ticker_cols = [c for c in feature_cols if c.startswith("ticker_")]
    non_ticker_cols = [c for c in feature_cols if not c.startswith("ticker_")]

    feature_input = row[non_ticker_cols].astype(float).to_dict()
    ticker_col_name = f"ticker_{ticker}"

    feature_df = _build_feature_frame(
        feature_input=feature_input,
        feature_cols=feature_cols,
        ticker_cols=ticker_cols,
        ticker_col_name=ticker_col_name,
    )
    # print(processed_df.iloc[-1].to_dict())
    prediction = model.predict(feature_df)
    pred_value = int(prediction[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feature_df)[0]
            confidence = float(proba[1]) if len(proba) > 1 else None
        except Exception:
            confidence = None

    return {
        "ticker": ticker,
        "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
        "prediction": pred_value,
        "label": "up" if pred_value == 1 else "down",
        "confidence": confidence,
        "model_artifact": str(model_path),
        "data": row.to_dict()
    }

# ----------------------------
# FastAPI app setup
# ----------------------------
app = FastAPI(title="Stock Prediction API")


# ----------------------------
# Request schema
# ----------------------------
class PredictionRequest(BaseModel):
    daily_return: float = Field(..., description="Percent change in close from previous day")
    high_low_spread_pct: float = Field(..., description="(High - Low) / Open")
    open_close_change_pct: float = Field(..., description="(Close - Open) / Open")
    volume_change: float = Field(..., description="Percent change in volume from previous day")
    ma_5: float = Field(..., description="5-day moving average of close")
    ma_10: float = Field(..., description="10-day moving average of close")
    volatility_5: float = Field(..., description="5-day rolling std dev of daily return")
    close_to_ma10_ratio: float = Field(..., description="Close / 10-day moving average")


# ----------------------------
# Optional response schema
# ----------------------------
class PredictionRequest(BaseModel):
    ticker: str


# ----------------------------
# Load model once at startup
# ----------------------------

@app.on_event("startup")
def startup_event():
    try:
        resources = load_backend_resources()
        tickers = get_available_tickers(resources["processed_df"])

        app.state.model = resources["model"]
        app.state.model_path = resources["model_path"]
        app.state.processed_df = resources["processed_df"]
        app.state.feature_cols = resources["feature_cols"]
        app.state.tickers = tickers
        app.state.startup_error = None

    except Exception as e:
        app.state.model = None
        app.state.model_path = None
        app.state.processed_df = None
        app.state.feature_cols = None
        app.state.startup_error = str(e)


# ----------------------------
# Health endpoint
# ----------------------------
@app.get("/health")
def health():
    if app.state.startup_error:
        return {
            "status": "error",
            "message": app.state.startup_error
        }

    return {
        "status": "ok",
        "model_path": str(app.state.model_path)
    }


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    print(request)
    if app.state.model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {app.state.startup_error}")

    requested_ticker = request.ticker
    # Convert validated request into a one-row DataFrame
    try:
        result = predict_for_ticker(
            ticker=requested_ticker,
            model=app.state.model,
            model_path=app.state.model_path,
            processed_df=app.state.processed_df,
            feature_cols=app.state.feature_cols,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict-aapl")
def predict_aapl():
    """
    Very basic test endpoint:
    - Uses hardcoded ticker AAPL
    - Pulls most recent processed row
    - Runs model prediction
    - Returns JSON response
    """
    res = get_available_tickers(app.state.processed_df)
    # Make sure startup succeeded
    if app.state.model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded: {app.state.startup_error}"
        )

    try:
        result = predict_for_ticker(
            ticker="AAPL",
            model=app.state.model,
            model_path=app.state.model_path,
            processed_df=app.state.processed_df,
            feature_cols=app.state.feature_cols,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
@app.get("/tickers")
def getTickers():
    return app.state.tickers