"""Evaluation metrics."""

from typing import Any, Dict


def compute_accuracy(predictions: list, labels: list) -> float:
    """Compute accuracy score."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions) if predictions else 0.0


def compute_f1(predictions: list, labels: list) -> Dict[str, float]:
    """Compute F1 score."""
    # TODO: Implement F1 computation
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
