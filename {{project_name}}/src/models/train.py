"""Model training module."""

import yaml
from pathlib import Path


def train():
    """Train model based on config."""
    config = load_params()
    # TODO: Implement training logic
    pass


def load_params():
    """Load parameters from params.yaml."""
    with open(Path(__file__).parent.parent.parent / "params.yaml") as f:
        return yaml.safe_load(f)
