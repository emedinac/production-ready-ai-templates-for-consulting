from mlflow.tracking import MlflowClient


def get_model_versions(model_name: str, client: MlflowClient | None = None):
    client = client or MlflowClient()
    return client.search_model_versions(f"name='{model_name}'")


def get_production_model(model_name: str, client: MlflowClient | None = None):
    versions = get_model_versions(model_name, client)

    production_versions = [v for v in versions if v.current_stage == "Production"]
    if not production_versions:
        return None

    return max(production_versions, key=lambda v: int(v.version))


def require_production_model(model_name: str):
    model = get_production_model(model_name)
    if model is None:
        raise ValueError(f"No Production model found for: {model_name}")
    return model
