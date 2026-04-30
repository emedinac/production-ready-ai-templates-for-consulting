from configs.validate import load_and_validate_config
from promotion.select import promote_best_model
from registry.promote import promote_to_production


def main():
    cfg = load_and_validate_config()

    # Step 1: pick best run
    model_uri = promote_best_model(cfg.tracking.mlflow.experiment_name)

    # Step 2: register + promote model
    version = promote_to_production(
        model_uri=model_uri, model_name=cfg.tracking.mlflow.experiment_name
    )

    print("Deployed model version:", version)


if __name__ == "__main__":
    main()
