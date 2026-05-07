"""Model training utilities."""

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)


def build_model(config):
    """Build a model based on config specifications.

    Args:
        config: FullConfig object with model, task, and training settings.

    Returns:
        Initialized model instance.

    Raises:
        NotImplementedError: If model type is not supported.
    """
    optimizer = config.training.optimizer.lower()
    learning_rate = float(config.training.learning_rate)
    epochs = int(config.training.epochs)

    if config.task.problem_type == "regression":
        if optimizer == "sgd":
            return SGDRegressor(
                max_iter=epochs, eta0=learning_rate, learning_rate="invscaling"
            )
        return LinearRegression()

    if config.task.problem_type in {"binary", "multiclass"}:
        if optimizer == "sgd":
            return SGDClassifier(
                max_iter=epochs,
                eta0=learning_rate,
                learning_rate="optimal",
                tol=1e-3,
            )
        return LogisticRegression(
            max_iter=max(epochs, 100), solver="lbfgs", multi_class="auto"
        )

    raise NotImplementedError(
        f"Task problem type '{config.task.problem_type}' is not supported"
    )
