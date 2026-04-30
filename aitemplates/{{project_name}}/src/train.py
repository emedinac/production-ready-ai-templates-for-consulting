import json
from pathlib import Path

import mlflow
import mlflow.pyfunc

from configs.validate import load_and_validate_config


class BaselineTextModel(mlflow.pyfunc.PythonModel):
    def __init__(self, label: int = 1):
        self.label = label

    def predict(self, context, model_input):
        return [self.label for _ in model_input]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def train(cfg):
    # Baseline model + metrics. Replace this block with real training.
    model = BaselineTextModel()

    metrics = {"accuracy": 0.91, "f1": 0.89}

    model_path = Path(cfg.output.model_path)
    metrics_path = Path(cfg.output.metrics_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(f"baseline model type={cfg.model.type}\n")
    write_json(metrics_path, metrics)

    return model, metrics


def main():
    cfg = load_and_validate_config()

    # -------------------------
    # MLflow tracking start
    # -------------------------
    mlflow.set_experiment(cfg.tracking.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": cfg.model.type,
                "dataset": cfg.data.source.type,
                "task": cfg.task.type,
                "seed": cfg.experiment.seed,
            }
        )

        model, metrics = train(cfg)

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cfg.output.metrics_path))
        mlflow.log_artifact(str(cfg.output.model_path))
        mlflow.pyfunc.log_model(artifact_path="model", python_model=model)

        print("Run logged to MLflow")


if __name__ == "__main__":
    main()
