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
from .train_utils import build_model


def _resolve_path(project_root: Path, path_str: str) -> Path:
    """Resolve relative paths from project root."""
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def train():
    """Train model based on configuration and preprocessed data."""
    config = load_config()
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = _resolve_path(project_root, config.data.processed_dir)
    model_path = _resolve_path(project_root, config.output.model_path)
    metrics_path = _resolve_path(project_root, config.output.metrics_path)

    processed_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

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

    model = build_model(config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    validation_metrics = compute_metrics(config, y_val, y_pred)

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
        "model_type": config.model.type,
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

    if feature_transformer is not None:
        artifact["feature_transformer"] = type(feature_transformer).__name__

    print("Training complete. Model artifact summary:")
    print(json.dumps(artifact, indent=2))
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return model_path, metrics_path


if __name__ == "__main__":
    train()
