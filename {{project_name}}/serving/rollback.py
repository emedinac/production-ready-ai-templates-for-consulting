from mlflow.tracking import MlflowClient


def rollback_to_previous(model_name: str):
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    prod_versions = sorted(
        (v for v in versions if v.current_stage == "Production"),
        key=lambda v: int(v.version),
        reverse=True,
    )

    if len(prod_versions) < 2:
        raise ValueError("No previous production model to rollback to")

    current = prod_versions[0]
    previous = prod_versions[1]

    client.transition_model_version_stage(
        name=model_name, version=current.version, stage="Archived"
    )

    client.transition_model_version_stage(
        name=model_name, version=previous.version, stage="Production"
    )

    print(f"Rolled back {model_name} to version {previous.version}")
