MIN_ACCURACY = 0.85
MIN_F1 = 0.80


def is_valid_run(metrics: dict[str, float]) -> bool:
    """
    Hard gate: must pass BEFORE promotion is even considered
    """

    if metrics.get("accuracy", 0.0) < MIN_ACCURACY:
        return False

    if metrics.get("f1", 0.0) < MIN_F1:
        return False

    return True
