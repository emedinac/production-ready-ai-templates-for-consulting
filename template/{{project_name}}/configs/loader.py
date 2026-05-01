import yaml
from pathlib import Path
from CHANGE_ME.configs.schema import FullConfig


def load_config(path: str = "params.yaml") -> FullConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return FullConfig(**raw)
