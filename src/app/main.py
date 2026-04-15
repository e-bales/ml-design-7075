from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predict import load_latest_model, predict

app = FastAPI(
    title="Stock Movement Prediction API",
    description="Predict whether a stock closing price will rise or fall on the next trading day.",
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


class PredictionRequest(BaseModel):
    features: dict


class PredictionResponse(BaseModel):
    predictions: list
    probabilities: list | None
    model_path: str


@app.get("/")
def health_check():
    return {"status": "ok", "model_directory": str(MODEL_DIR)}


@app.post("/predict", response_model=PredictionResponse)
def predict_stock(request: PredictionRequest):
    try:
        model, model_path = load_latest_model(MODEL_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not isinstance(request.features, dict):
        raise HTTPException(status_code=422, detail="Request features must be a JSON object of numeric values.")

    result = predict(model, request.features)
    return {
        "predictions": result["predictions"],
        "probabilities": result["probabilities"],
        "model_path": str(model_path),
    }
