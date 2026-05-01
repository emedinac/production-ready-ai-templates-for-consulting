"""Batch experiment runner."""

from typing import List, Dict, Any
import itertools


def run_batch(configs: List[Dict[str, Any]]):
    """Run multiple config experiments safely."""
    results = []
    for config in configs:
        # TODO: Execute each config
        results.append(config)
    return results


def generate_configs(param_grid: Dict[str, List]) -> List[Dict]:
    """Generate configs from parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))
