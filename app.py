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

# ----------------------------
# Your existing model-loading logic
# ----------------------------
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
class PredictionResponse(BaseModel):
    prediction: int
    model_artifact: str


# ----------------------------
# Load model once at startup
# ----------------------------

try:
    model, artifact_dir = load_latest_model(MODELS_DIR)
except Exception as e:
    model = None
    artifact_dir = None
    startup_error = str(e)
else:
    startup_error = None


# ----------------------------
# Health endpoint
# ----------------------------
@app.get("/health")
def health():
    if model is None:
        return {
            "status": "error",
            "message": startup_error,
        }

    return {
        "status": "ok",
        "model_artifact": str(artifact_dir),
    }


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {startup_error}")

    # Convert validated request into a one-row DataFrame
    input_df = pd.DataFrame([request.model_dump()])

    try:
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        prediction=int(prediction[0]),
        model_artifact=str(artifact_dir),
    )