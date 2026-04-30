import mlflow
from promotion.rules import is_valid_run


def get_best_run(experiment_name: str, metric_name: str = "accuracy"):
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(experiment.experiment_id)

    best_run = None
    best_score = float("-inf")

    for run in runs:
        metrics = run.data.metrics

        if not is_valid_run(metrics):
            continue

        score = metrics.get(metric_name)
        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_run = run

    return best_run


def promote_best_model(experiment_name: str):
    best_run = get_best_run(experiment_name)

    if not best_run:
        raise ValueError(f"No valid model found for promotion in experiment: {experiment_name}")

    model_uri = f"runs:/{best_run.info.run_id}/model"

    print(f"Promoting model: {model_uri}")

    return model_uri
