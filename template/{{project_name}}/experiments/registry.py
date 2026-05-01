"""MLflow interaction wrapper."""

from typing import Any, Dict, Optional
import mlflow


def start_run(run_name: Optional[str] = None):
    """Start an MLflow run."""
    return mlflow.start_run(run_name=run_name)


def log_param(key: str, value: Any):
    """Log a parameter to MLflow."""
    mlflow.log_param(key, value)


def log_metric(key: str, value: float):
    """Log a metric to MLflow."""
    mlflow.log_metric(key, value)


def log_model(model, artifact_path: str):
    """Log a model to MLflow."""
    mlflow.sklearn.log_model(model, artifact_path)
