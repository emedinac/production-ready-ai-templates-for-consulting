"""Optional config templates."""

from typing import Dict, Any

TEMPLATES = {
    "baseline": {
        "model": {"type": "linear"},
        "experiment": {"seed": 42},
    },
    "advanced": {
        "model": {"type": "transformer"},
        "experiment": {"seed": 123},
    },
}


def get_template(name: str) -> Dict[str, Any]:
    """Get a config template by name."""
    return TEMPLATES.get(name, {})
