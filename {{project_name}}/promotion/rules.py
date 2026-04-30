def is_valid_run(metrics: dict) -> bool:
    """
    Hard gate: must pass BEFORE promotion is even considered
    """

    if metrics["accuracy"] < 0.85:
        return False

    if metrics["f1"] < 0.80:
        return False

    return True
