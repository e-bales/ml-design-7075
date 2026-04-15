from pathlib import Path

import numpy as np
import pandas as pd

from src.models.train import train_model


def test_train_model_logs_and_saves_model(tmp_path: Path):
    X = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature_2": [1.0, 0.9, 0.8, 0.7, 0.6],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])
    model_dir = tmp_path / "models"

    model, model_path, run_id, metrics = train_model(
        X=X,
        y=y,
        model_dir=model_dir,
        run_name="test_run",
    )

    assert model_path.exists()
    assert run_id is not None
    assert metrics["accuracy"] >= 0.0
    assert hasattr(model, "predict")
