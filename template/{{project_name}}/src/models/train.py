"""Model training module."""

import json
from pathlib import Path

import joblib
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from ...configs.loader import load_config


def train() -> Path:
    config = load_config()

    processed_dir = Path(config.data.processed_dir)
    train_file = processed_dir / "train.csv"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Processed training data not found: {train_file}. Run preprocess first."
        )

    # Load data
    df = pd.read_csv(train_file)

    if config.task.type == "text_classification":
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Expected 'text' and 'label' columns in training data")
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]
    else:
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
    artifacts_dir = Path(config.output.model_path).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifacts_dir / "model.joblib")
    if config.task.type == "text_classification":
        joblib.dump(vectorizer, artifacts_dir / "vectorizer.joblib")

    model_card = {
        "model_type": config.model.type,
        "task": config.task.type,
        "training": config.training.model_dump(),
        "features": "text_vectorized"
        if config.task.type == "text_classification"
        else list(X.columns),
    }

    (artifacts_dir / "model.yaml").write_text(
        yaml.safe_dump(model_card, sort_keys=True)
    )

    metrics_path = Path(config.output.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    return artifacts_dir


def load_params():
    with open(Path("params.yaml")) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    train()
