"""OPTIONAL: Advisory eval agent - not decision authority."""

from typing import Dict, Any


def evaluate(project_state: Dict[str, Any]) -> Dict[str, str]:
    """Provide advisory evaluation feedback."""
    # TODO: Implement eval agent logic
    return {"recommendation": "proceed", "notes": ""}
