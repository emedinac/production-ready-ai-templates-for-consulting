"""Utility functions."""

from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    import yaml
    from pathlib import Path

    with open(Path(path)) as f:
        return yaml.safe_load(f)
