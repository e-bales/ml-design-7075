from argparse import ArgumentParser
from pathlib import Path

from src.data.feature_engineering import save_processed_features


def main() -> None:
    parser = ArgumentParser(description="Create processed feature CSVs from raw Alpha Vantage data.")
    parser.add_argument(
        "raw_file",
        help="Raw CSV file path output from Alpha Vantage, e.g. data/raw/AAPL_daily.csv",
    )
    parser.add_argument(
        "--ticker",
        help="Ticker symbol to assign if not present in raw file.",
        default=None,
    )
    parser.add_argument(
        "--processed-dir",
        help="Directory where processed feature CSVs should be stored.",
        default="data/processed",
    )
    args = parser.parse_args()

    raw_file = Path(args.raw_file)
    if not raw_file.exists():
        raise SystemExit(f"Raw file not found: {raw_file}")

    processed_dir = Path(args.processed_dir)
    output_file = save_processed_features(raw_file, processed_dir, ticker=args.ticker)
    print(f"Processed features saved to: {output_file}")


if __name__ == "__main__":
    main()
