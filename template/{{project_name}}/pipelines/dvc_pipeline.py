"""DVC pipeline helpers."""

from pathlib import Path
import subprocess


def run_stage(stage_name: str):
    """Run a DVC stage by name."""
    subprocess.run(["dvc", "repro", stage_name], check=True)


def list_stages() -> list[str]:
    """List available pipeline stages."""
    # TODO: Parse dvc.yaml
    return ["ingest", "preprocess", "train", "evaluate"]
