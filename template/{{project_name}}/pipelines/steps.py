"""Reusable pipeline steps."""

from typing import Any, Dict


def validate_input(data: Any, schema: Dict) -> bool:
    """Validate input data against schema."""
    # TODO: Implement validation
    return True


def cache_output(output: Any, path: str):
    """Cache pipeline output."""
    # TODO: Implement caching
    pass
