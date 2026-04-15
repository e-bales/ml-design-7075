from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .feature_engineering import FEATURE_COLUMNS


def load_dataset(data_dir: Path, filename: str) -> pd.DataFrame:
    """Load a CSV dataset from the data directory."""
    csv_path = data_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
):
    """Separate features and target, using a fixed feature set when available."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in dataframe columns")

    selected_features = feature_columns or FEATURE_COLUMNS
    selected_features = [col for col in selected_features if col in df.columns]
    if not selected_features:
        selected_features = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns.tolist()

    X = df[selected_features].copy()
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)

    if X.empty:
        raise ValueError("No numeric feature columns remain after dropping non-numeric metadata columns.")

    y = df[target_column]
    return X, y
