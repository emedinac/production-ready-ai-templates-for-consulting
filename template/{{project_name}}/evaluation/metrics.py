"""Evaluation metrics."""

import json
from pathlib import Path
from typing import Dict


def compute_accuracy(predictions: list, labels: list) -> float:
    """Compute accuracy score."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions) if predictions else 0.0


def compute_f1(predictions: list, labels: list) -> Dict[str, float]:
    """Compute F1 score."""
    # TODO: Implement F1 computation
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def write_evaluation_metrics() -> Path:
    """Write placeholder evaluation metrics for the DVC stage."""
    output_path = Path("results") / "eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"accuracy": 0.0, "f1": 0.0}, indent=2) + "\n")
    return output_path


if __name__ == "__main__":
    write_evaluation_metrics()
