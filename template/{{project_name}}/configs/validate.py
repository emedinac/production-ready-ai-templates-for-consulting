from pathlib import Path

import yaml
from CHANGE_ME.configs.schema import FullConfig


CONFIG_PATH = Path("params.yaml")


def load_and_validate_config(path: Path = CONFIG_PATH) -> FullConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw_cfg = yaml.safe_load(f)

    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return FullConfig.model_validate(raw_cfg)


if __name__ == "__main__":
    cfg = load_and_validate_config()
    print("Config is valid.")
