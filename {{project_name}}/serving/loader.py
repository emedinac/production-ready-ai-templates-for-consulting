import mlflow.pyfunc


def load_production_model(model_name: str):
    # Always pulls latest Production model
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
