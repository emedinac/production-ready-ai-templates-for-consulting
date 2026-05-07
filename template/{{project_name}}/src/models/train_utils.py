"""Model training utilities."""

from sklearn.linear_model import LinearRegression, LogisticRegression


def build_model(config):
    """Build a model based on config specifications.

    Args:
        config: FullConfig object with model and task specifications.

    Returns:
        Initialized model instance.

    Raises:
        NotImplementedError: If model type is not supported.
    """
    if config.model.type == "linear":
        if config.task.problem_type == "regression":
            return LinearRegression()
        return LogisticRegression(max_iter=5_000, solver="lbfgs", multi_class="auto")

    raise NotImplementedError(
        f"Model type '{config.model.type}' is not implemented in this training script"
    )
