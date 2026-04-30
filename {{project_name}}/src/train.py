from configs.validate import load_and_validate_config


def main():
    # HARD REQUIREMENT: always load config first
    cfg = load_and_validate_config()

    # Example usage
    print(f"Running experiment: {cfg.experiment.name}")
    print(f"Model type: {cfg.model.type}")
    print(f"Dataset source: {cfg.data.source.type}")

    # Your training logic starts here
    # No hardcoded values allowed


if __name__ == "__main__":
    main()
