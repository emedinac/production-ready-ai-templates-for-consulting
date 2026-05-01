"""Data preprocessing module."""

import yaml
from pathlib import Path


def preprocess():
    """Preprocess data based on config."""
    config = load_params()
    # TODO: Implement preprocessing logic
    pass


def load_params():
    """Load parameters from params.yaml."""
    with open(Path(__file__).parent.parent / "params.yaml") as f:
        return yaml.safe_load(f)
