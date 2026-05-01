"""Evaluation reporting."""

from typing import Dict, Any
from pathlib import Path


def generate_report(results: Dict[str, Any], output_path: str):
    """Generate evaluation report."""
    # TODO: Implement report generation
    pass


def save_metrics(results: Dict[str, Any]):
    """Save metrics to file."""
    import json

    path = Path("metrics.json")
    path.write_text(json.dumps(results, indent=2))
