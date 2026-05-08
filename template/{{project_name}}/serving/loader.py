import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


def load_model(model_uri: str):
    return mlflow.pyfunc.load_model(model_uri)


def production_model_uri(model_name: str) -> str:
    return f"models:/{model_name}/latest"


def load_production_model(model_name: str):
    return load_model(production_model_uri(model_name))


def load_latest_from_experiment(experiment_name: str):
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)

    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found for experiment: {experiment_name}")

    run_id = runs[0].info.run_id

    model_uri = f"runs:/{run_id}/model"

    return mlflow.pyfunc.load_model(model_uri)
