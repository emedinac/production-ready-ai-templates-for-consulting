from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(config, y_true, y_pred) -> dict:
    """Compute evaluation metrics based on task type.

    Args:
        config: FullConfig object with task specifications.
        y_true: Ground truth labels/values.
        y_pred: Predicted labels/values.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}
    if config.task.problem_type == "regression":
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
    else:
        average = "binary" if config.task.problem_type == "binary" else "macro"
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        )
    return metrics


def write_evaluation_metrics() -> Path:
    project_root = Path(__file__).resolve().parents[1]

    output_path = project_root / "results/eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps({"accuracy": 0.0, "f1": 0.0}, indent=2))

    return output_path


if __name__ == "__main__":
    write_evaluation_metrics()
