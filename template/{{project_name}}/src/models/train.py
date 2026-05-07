from pathlib import Path
import json

import joblib
import pandas as pd

from ...configs.loader import load_config
from ...evaluation.metrics import compute_metrics
from ...src.data.features import (
    prepare_text_classification_data,
    prepare_tabular_data,
)
from .train_utils import build_model, train_transformer
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


def _resolve_path(project_root: Path, path_str: str) -> Path:
    """Resolve relative paths from project root."""
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _experiment_results_dir(repo_root: Path, experiment_name: str) -> Path:
    return repo_root / "results" / experiment_name


def train():
    """Train model based on configuration and preprocessed data."""
    config = load_config()
    repo_root = Path(__file__).resolve().parents[3]

    processed_dir = _resolve_path(repo_root, config.data.processed_dir)
    model_path = _resolve_path(repo_root, config.output.model_path)
    metrics_path = _resolve_path(repo_root, config.output.metrics_path)

    processed_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    results_experiment_dir = _experiment_results_dir(repo_root, config.experiment.name)
    results_model_dir = results_experiment_dir / "models"
    results_metrics_dir = results_experiment_dir / "metrics"
    results_model_dir.mkdir(parents=True, exist_ok=True)
    results_metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    print(f"Loaded train samples: {len(train_df)}")
    print(f"Loaded validation samples: {len(val_df)}")

    if len(train_df) == 0:
        raise ValueError("Train dataset is empty")
    if len(val_df) == 0:
        raise ValueError("Validation dataset is empty")

    if config.task.type == "text_classification":
        X_train, X_val, y_train, y_val, feature_transformer = (
            prepare_text_classification_data(train_df, val_df, config)
        )
    else:
        X_train, X_val, y_train, y_val, feature_transformer = prepare_tabular_data(
            train_df, val_df, config
        )

    bundle = build_model(config)

    if bundle.type == "transformer":
        bundle.model = train_transformer(bundle, train_df, val_df, config)
    else:
        bundle.model.fit(X_train, y_train)

    # prediction MUST use bundle.model
    y_pred = bundle.model.predict(X_val)
    validation_metrics = compute_metrics(config, y_val, y_pred)

    print("\n=== Data Distribution ===")
    print("Train distribution:", dict(Counter(y_train)))
    print("Validation distribution:", dict(Counter(y_val)))
    print("Predictions distribution:", dict(Counter(y_pred)))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("=== End Diagnostics ===")

    metrics = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "task": config.task.type,
        "problem_type": config.task.problem_type,
        "training": {
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epochs,
            "scheduler": config.training.scheduler.type,
        },
        "validation": validation_metrics,
    }

    artifact = {
        "config_model_type": config.model.type,
        "trained_model_class": type(bundle.model).__name__,
        "feature_preprocessor": type(feature_transformer).__name__
        if feature_transformer
        else None,
        "task": config.task.type,
        "problem_type": config.task.problem_type,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "training_config": {
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epochs,
        },
    }

    print("Training complete. Model artifact summary:")
    print(json.dumps(artifact, indent=2))
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))

    joblib.dump(bundle.model, model_path)
    joblib.dump(bundle.model, results_model_dir / model_path.name)

    metrics_path.write_text(json.dumps(metrics, indent=2))
    (results_metrics_dir / metrics_path.name).write_text(json.dumps(metrics, indent=2))

    return model_path, metrics_path


if __name__ == "__main__":
    train()
