from pathlib import Path

import pandas as pd

from src.data.feature_engineering import FEATURE_COLUMNS
from src.data.load_data import split_features_target
from src.data.pipeline import ingest_data


def test_ingest_data_creates_versioned_file(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    data = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "open": [100.0, 101.5],
            "close": [101.0, 100.5],
            "target": [1, 0],
        }
    )
    raw_file = raw_dir / "stock_data.csv"
    data.to_csv(raw_file, index=False)

    expected_columns = {
        "date": "datetime",
        "open": "numeric",
        "close": "numeric",
        "target": "numeric",
    }

    df, version_path, metadata_path = ingest_data(raw_dir, "stock_data.csv", processed_dir, expected_columns)

    assert df.shape == (2, 4)
    assert version_path.exists()
    assert metadata_path.exists()
    metadata = metadata_path.read_text(encoding="utf-8")
    assert "hash" in metadata


def test_split_features_target_uses_engineered_features():
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0],
            "prev_close": [99.0, 101.0, 102.0],
            "daily_return": [0.01, 0.0099, 0.0098],
            "ma_3": [100.0, 101.5, 102.5],
            "ma_diff": [1.0, 0.5, 0.5],
            "sentiment_compound": [0.1, -0.2, 0.0],
            "sentiment_pos": [0.2, 0.1, 0.0],
            "sentiment_neu": [0.7, 0.8, 0.9],
            "sentiment_neg": [0.1, 0.1, 0.1],
            "target": [1, 0, 1],
        }
    )
    X, y = split_features_target(df, target_column="target")

    assert list(X.columns) == [col for col in FEATURE_COLUMNS if col in df.columns]
    assert len(X.columns) == 8
    assert list(y) == [1, 0, 1]
