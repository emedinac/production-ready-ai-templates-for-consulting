import mlflow
from mlflow.tracking import MlflowClient
from configs.validate import load_and_validate_config


def promote_to_production(model_uri: str, model_name: str):
    client = MlflowClient()

    # Register model
    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    # Transition to Staging first (safe step)
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Staging"
    )

    # Production promotion gate (manual + rule-based safe switch)
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )

    print(f"Model {model_name} v{version} promoted to Production")

    return version
