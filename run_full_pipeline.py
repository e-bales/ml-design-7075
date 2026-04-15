from pathlib import Path

import pandas as pd

from src.data.alphavantage import save_daily_adjusted
from src.data.feature_engineering import save_processed_features
from src.data.load_data import split_features_target
from src.models.train import train_model


def main(symbol: str) -> None:
    root = Path(__file__).resolve().parent
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"

    raw_file = None
    try:
        print(f"Fetching raw daily data for {symbol} from Alpha Vantage...")
        raw_file = save_daily_adjusted(symbol, raw_dir)
        print(f"Saved raw file: {raw_file}")
    except Exception as exc:
        sample_file = raw_dir / f"{symbol.upper()}_daily_sample.csv"
        print("Alpha Vantage fetch failed:", exc)
        if sample_file.exists():
            raw_file = sample_file
            print(f"Using local fallback sample data: {sample_file}")
        else:
            raise SystemExit(
                "Alpha Vantage fetch failed and no local sample data is available. "
                "Please wait and retry later, or add a raw CSV file to data/raw/."
            )

    print("Building processed features...")
    processed_file = save_processed_features(raw_file, processed_dir, ticker=symbol)
    print(f"Saved processed feature file: {processed_file}")

    print("Loading processed features and training the model...")
    df = pd.read_csv(processed_file)
    X, y = split_features_target(df, "target")
    model, model_path, run_id, metrics = train_model(X, y, models_dir)

    print(f"Training complete. Model saved to: {model_path}")
    print(f"Registry updated with run_id: {run_id}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch data, process features, and train the stock model.")
    parser.add_argument("symbol", help="Ticker symbol to fetch from Alpha Vantage, e.g. AAPL")
    args = parser.parse_args()
    main(args.symbol)
