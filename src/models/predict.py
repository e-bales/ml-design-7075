from pathlib import Path
from typing import Dict

import pandas as pd
from joblib import load


def get_latest_model_path(model_dir: Path) -> Path:
    model_dir = model_dir.resolve()
    model_files = sorted(
        [path for path in model_dir.glob("*.joblib") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    return model_files[0]


def load_latest_model(model_dir: Path):
    model_path = get_latest_model_path(model_dir)
    return load(model_path), model_path


def predict(model, features: Dict[str, float]):
    input_df = pd.DataFrame([features])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)
    prediction = model.predict(input_df)
    return {
        "predictions": prediction.tolist(),
        "probabilities": probabilities.tolist() if probabilities is not None else None,
    }
