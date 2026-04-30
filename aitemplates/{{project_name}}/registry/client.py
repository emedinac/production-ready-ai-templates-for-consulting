from mlflow.tracking import MlflowClient


def get_model_versions(model_name: str):
    client = MlflowClient()
    return client.search_model_versions(f"name='{model_name}'")


def get_production_model(model_name: str):
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    for v in versions:
        if v.current_stage == "Production":
            return v

    return None
