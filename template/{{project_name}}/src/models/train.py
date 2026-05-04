"""Model training module."""

import json
from pathlib import Path

import yaml

from ...configs.loader import load_config


def project_root() -> Path:
    """Return the generated package root from the current project directory."""
    package_name = __package__.split(".", 1)[0]
    return Path.cwd() / package_name


def train() -> Path:
    """Create deterministic placeholder model artifacts and metrics."""
    config = load_config()
    root = project_root()
    processed_dir = root / "data" / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found: {processed_dir}. Run the preprocess stage first."
        )

    artifacts_dir = root / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_card = {
        "model_type": config.model.type,
        "task": config.task.type,
        "training": config.training.model_dump(),
    }
    (artifacts_dir / "model.yaml").write_text(
        yaml.safe_dump(model_card, sort_keys=True)
    )

    metrics = {"accuracy": 0.0, "f1": 0.0}
    metrics_path = root / "models" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return artifacts_dir


def load_params():
    """Load parameters from params.yaml."""
    with open(Path("params.yaml")) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    train()
