"""Promotion rules - hard-coded promotion logic."""

from typing import Dict, Any, Optional


def can_promote(model_info: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    """Check if model meets promotion thresholds."""
    score = model_info.get("score", 0.0)
    return score >= thresholds.get("min_score", 0.0)


def get_promotion_stage(model_info: Dict[str, Any]) -> str:
    """Determine promotion stage for model."""
    # TODO: Implement stage logic
    return "staging"
