from promotion.select import promote_best_model
from configs.validate import load_and_validate_config


def main():
    cfg = load_and_validate_config()

    model_uri = promote_best_model(cfg.tracking.mlflow.experiment_name)

    print("Selected model:", model_uri)


if __name__ == "__main__":
    main()
