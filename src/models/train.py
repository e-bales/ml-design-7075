from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .registry import register_model_version


def configure_mlflow_tracking(tracking_dir: Path = Path("mlruns")) -> None:
    tracking_dir = tracking_dir.resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("stock_movement_prediction")


def build_pipeline() -> Pipeline:
    """Build a reusable sklearn pipeline for stock movement prediction."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42, n_estimators=100)),
        ]
    )


def train_model(
    X,
    y,
    model_dir: Path,
    run_name: str = "stock_movement_run",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Path, str, Dict[str, float]]:
    """Train a model, log the experiment with MLflow, and version the model artifact."""
    configure_mlflow_tracking()
    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.sklearn.autolog()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = build_pipeline()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))
        report = classification_report(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")

        model_filename = f"stock_movement_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_path = model_dir / model_filename
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")

        register_model_version(
            model_dir=model_dir,
            model_filename=model_filename,
            run_id=run.info.run_id,
            metrics={"accuracy": accuracy},
            parameters=model.get_params(),
        )

    return model, model_path, run.info.run_id, {"accuracy": accuracy}
