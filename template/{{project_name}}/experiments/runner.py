"""Experiment runner."""

from typing import Dict, Any, Optional
import mlflow


def run_experiment(config: Dict[str, Any], run_name: Optional[str] = None):
    """Execute a single experiment run."""
    with mlflow.start_run(run_name=run_name):
        # TODO: Implement experiment execution
        pass


def run_training(config: Dict[str, Any]):
    """Run training with config."""
    # TODO: Implement training
    pass
