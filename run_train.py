from pathlib import Path

import pandas as pd

from src.data.load_data import split_features_target
from src.models.train import train_model

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed" / "versions"
MODELS_DIR = ROOT / "models"
TARGET_COLUMN = "target"

csv_files = sorted(PROCESSED_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
if not csv_files:
    raise SystemExit(
        "No processed CSV file found in data/processed/versions/. "
        "Run your ingestion pipeline first and save a processed dataset there."
    )

latest_csv = csv_files[0]
print(f"Loading processed dataset: {latest_csv}")
df = pd.read_csv(latest_csv)
X, y = split_features_target(df, TARGET_COLUMN)
print(f"Training model on {len(df)} rows with target '{TARGET_COLUMN}'.")
model, model_path, run_id, metrics = train_model(X, y, MODELS_DIR)
print(f"Model saved to: {model_path}")
print(f"Model registry updated with run_id: {run_id}")
print(f"Metrics: {metrics}")
