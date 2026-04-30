from configs.validate import load_and_validate_config
import json
from pathlib import Path
import mlflow


def train(cfg):
    # fake model + metrics (replace with real training later)
    model = {"type": cfg.model.type}

    metrics = {"accuracy": 0.91, "f1": 0.89}

    # save artifacts
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "w") as f:
        f.write(str(model))

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    return model, metrics


def main():
    cfg = load_and_validate_config()

    # -------------------------
    # MLflow tracking start
    # -------------------------
    mlflow.set_experiment(cfg.tracking.mlflow.experiment_name)

    with mlflow.start_run():
        # log config (important for reproducibility)
        mlflow.log_params({
            "model_type": cfg.model.type,
            "dataset": cfg.data.source.type,
            "task": cfg.task.type,
            "seed": cfg.experiment.seed,
        })

        model, metrics = train(cfg)

        # log metrics
        mlflow.log_metrics(metrics)

        # log artifacts
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("models/model.pkl")

        print("Run logged to MLflow")


if __name__ == "__main__":
    main()
