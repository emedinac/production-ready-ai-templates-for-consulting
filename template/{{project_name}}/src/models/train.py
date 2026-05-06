from pathlib import Path
import json
import pandas as pd

from ...configs.loader import load_config


def train():
    config = load_config()

    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data/processed"
    artifacts_dir = project_root / "models/artifacts"
    metrics_path = project_root / "models/metrics.json"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    print(f"Loaded train samples: {len(train_df)}")
    print(f"Loaded validation samples: {len(val_df)}")

    if len(train_df) == 0:
        raise ValueError("Train dataset is empty")
    if len(val_df) == 0:
        raise ValueError("Validation dataset is empty")

    # "Fake model" placeholder (replace later with real training)
    model_summary = {
        "type": "baseline_stub",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }
    print("Training complete. Model summary:")
    print(json.dumps(model_summary, indent=2))

    # Save model artifact
    (artifacts_dir / "model.json").write_text(json.dumps(model_summary, indent=2))

    # Save metrics
    metrics = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }
    print("Training metrics:")
    print(json.dumps(metrics, indent=2))

    metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
