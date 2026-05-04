"""Model training module."""

import json
from pathlib import Path

import joblib
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from ...configs.loader import load_config


def project_root() -> Path:
    package_name = __package__.split(".", 1)[0]
    return Path.cwd() / package_name


def train() -> Path:
    config = load_config()
    root = project_root()

    processed_dir = root / "data" / "processed"
    train_file = processed_dir / "train.csv"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Processed training data not found: {train_file}. Run preprocess first."
        )

    # Load data
    df = pd.read_csv(train_file)

    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in training data")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    preds = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="weighted")),
    }

    # Save artifacts
    artifacts_dir = root / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifacts_dir / "model.joblib")

    model_card = {
        "model_type": config.model.type,
        "task": config.task.type,
        "training": config.training.model_dump(),
        "features": list(X.columns),
    }

    (artifacts_dir / "model.yaml").write_text(
        yaml.safe_dump(model_card, sort_keys=True)
    )

    metrics_path = root / "models" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    return artifacts_dir


def load_params():
    with open(Path("params.yaml")) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    train()
