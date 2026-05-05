"""Data preprocessing module."""

from pathlib import Path
import json
import yaml

from ...configs.loader import load_config


def preprocess() -> Path:
    config = load_config()

    # CONFIG DRIVEN PATH (not hardcoded)
    output_dir = Path(config.data.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- DATA ARTIFACT ----
    metadata = {
        "task": config.task.type,
        "problem_type": config.task.problem_type,
        "source_type": config.data.source.type,
        "split": config.data.split.model_dump(),
    }

    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=True))

    # ---- OPTIONAL: REAL METRICS ONLY IF COMPUTED ----
    metrics = {
        "status": "completed",
        "num_files": len(list(output_dir.glob("*"))),
    }

    metrics_dir = Path(config.artifacts.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "preprocess.json").write_text(json.dumps(metrics, indent=2))

    return output_dir


if __name__ == "__main__":
    preprocess()
