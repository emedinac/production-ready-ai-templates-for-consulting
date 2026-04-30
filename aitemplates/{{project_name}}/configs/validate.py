# configs/validate.py

import yaml
from pathlib import Path
from configs.schema import FullConfig


CONFIG_PATH = Path("params.yaml")


def load_and_validate_config(path: Path = CONFIG_PATH) -> FullConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # This is the critical line: validation happens here
    config = FullConfig(**raw_cfg)

    return config


if __name__ == "__main__":
    cfg = load_and_validate_config()
    print("Config is valid.")
