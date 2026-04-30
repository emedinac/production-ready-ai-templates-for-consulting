import mlflow.pyfunc


def load_model(model_uri: str):
    return mlflow.pyfunc.load_model(model_uri)


def production_model_uri(model_name: str) -> str:
    return f"models:/{model_name}/Production"


def load_production_model(model_name: str):
    return load_model(production_model_uri(model_name))
