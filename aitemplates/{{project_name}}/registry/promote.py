import mlflow
from mlflow.tracking import MlflowClient


def promote_to_production(model_uri: str, model_name: str):
    client = MlflowClient()

    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Staging"
    )

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Model {model_name} v{version} promoted to Production")

    return version
