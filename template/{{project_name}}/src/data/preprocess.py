"""Data preprocessing module."""

from pathlib import Path

import yaml

from ...configs.loader import load_config


def project_root() -> Path:
    """Return the generated package root from the current project directory."""
    package_name = __package__.split(".", 1)[0]
    return Path.cwd() / package_name


def preprocess() -> Path:
    """Create a deterministic placeholder processed dataset."""
    config = load_config()
    output_dir = project_root() / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task": config.task.type,
        "problem_type": config.task.problem_type,
        "source_type": config.data.source.type,
        "split": config.data.split.model_dump(),
    }
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=True))
    return output_dir


def load_params():
    """Load parameters from params.yaml."""
    with open(Path("params.yaml")) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    preprocess()
