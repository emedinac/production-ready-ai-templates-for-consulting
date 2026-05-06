from pathlib import Path
import json
import pandas as pd

from ...configs.loader import load_config


def train():
    config = load_config()

    # PATHS
    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data/processed"
    artifacts_dir = project_root / "models/artifacts"
    checkpoints_dir = project_root / "models/checkpoints"
    metrics_path = project_root / "models/metrics.json"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # LOAD PROCESSED DATA (NOT RAW)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")

    # DUMMY TRAINING (replace later)
    model = {
        "num_train_samples": len(train_df),
        "num_val_samples": len(val_df),
    }

    # SAVE ARTIFACTS
    (artifacts_dir / "model.json").write_text(json.dumps(model, indent=2))

    # SAVE METRICS
    metrics = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
