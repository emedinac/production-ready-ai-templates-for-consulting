"""Data drift detection."""

from typing import Dict, Any, List
import numpy as np


def detect_drift(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Detect data drift between reference and current distributions."""
    # TODO: Implement drift detection
    return {"drift_score": 0.0, "status": "no_drift"}


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute basic statistics."""
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }
