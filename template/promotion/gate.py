"""CI promotion gate helper."""

from typing import Dict, Any
from promotion.rules import can_promote


def run_gate(model_info: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    """Run promotion gate checks."""
    return can_promote(model_info, thresholds)


def load_thresholds(path: str = "promotion/thresholds.yaml") -> Dict[str, float]:
    """Load promotion thresholds."""
    import yaml
    from pathlib import Path

    with open(Path(path)) as f:
        return yaml.safe_load(f)
