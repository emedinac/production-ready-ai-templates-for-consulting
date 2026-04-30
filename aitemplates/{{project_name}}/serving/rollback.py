import mlflow
from mlflow.tracking import MlflowClient


def rollback_to_previous(model_name: str):
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    # get production versions sorted
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if len(prod_versions) < 2:
        raise ValueError("No previous production model to rollback to")

    # demote current production
    current = prod_versions[-1]
    previous = prod_versions[-2]

    client.transition_model_version_stage(
        name=model_name, version=current.version, stage="Archived"
    )

    client.transition_model_version_stage(
        name=model_name, version=previous.version, stage="Production"
    )

    print(f"Rolled back to version {previous.version}")
