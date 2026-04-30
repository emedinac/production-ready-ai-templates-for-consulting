from configs.validate import load_and_validate_config
import json
from pathlib import Path


def train(cfg):
    # simulate training result
    model = {"type": cfg.model.type}

    metrics = {"accuracy": 0.91, "f1": 0.89}

    # save model artifact
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "w") as f:
        f.write(str(model))

    # save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


def main():
    cfg = load_and_validate_config()

    print(f"Running: {cfg.experiment.name}")
    metrics = train(cfg)

    print("Training complete:", metrics)


if __name__ == "__main__":
    main()
