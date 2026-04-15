from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def validate_dataframe(df: pd.DataFrame, expected_columns: Dict[str, str]) -> None:
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    validation_errors = []
    for column, dtype in expected_columns.items():
        if dtype == "numeric" and not pd.api.types.is_numeric_dtype(df[column]):
            validation_errors.append(f"Column '{column}' must be numeric.")
        if dtype == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                try:
                    pd.to_datetime(df[column], errors="raise")
                except Exception as exc:
                    validation_errors.append(f"Column '{column}' must be datetime: {exc}")

    if df.isna().any(axis=None):
        missing = df.isna().sum()
        nonzero = missing[missing > 0].to_dict()
        validation_errors.append(f"Missing values found: {nonzero}")

    if validation_errors:
        raise ValueError("Data validation failed:\n" + "\n".join(validation_errors))


def save_versioned_data(
    df: pd.DataFrame,
    processed_dir: Path,
    source_file: Path,
    data_hash: str,
    schema: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    processed_dir = processed_dir.resolve()
    versions_dir = processed_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_name = f"{source_file.stem}_{timestamp}_{data_hash[:8]}.csv"
    version_path = versions_dir / version_name
    df.to_csv(version_path, index=False)

    metadata = {
        "source_file": str(source_file.resolve()),
        "created_at": timestamp,
        "hash": data_hash,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "schema": schema or {},
    }
    metadata_path = versions_dir / f"{version_name}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return version_path, metadata_path


def ingest_data(
    raw_dir: Path,
    filename: str,
    processed_dir: Path,
    expected_columns: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    raw_path = Path(raw_dir) / filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    if expected_columns is not None:
        validate_dataframe(df, expected_columns)

    data_hash = compute_dataframe_hash(df)
    version_path, metadata_path = save_versioned_data(
        df=df,
        processed_dir=processed_dir,
        source_file=raw_path,
        data_hash=data_hash,
        schema=expected_columns,
    )
    return df, version_path, metadata_path
