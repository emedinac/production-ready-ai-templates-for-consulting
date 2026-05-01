"""Config generator - NOT system generator."""

from typing import Dict, Any, List
import yaml
from pathlib import Path


def generate_config(template: str, output_path: str, params: Dict[str, Any]):
    """Generate a config from template."""
    # TODO: Implement config generation
    pass


def load_template(name: str) -> Dict[str, Any]:
    """Load a config template."""
    templates_dir = Path(__file__).parent / "templates"
    with open(templates_dir / f"{name}.yaml") as f:
        return yaml.safe_load(f)
