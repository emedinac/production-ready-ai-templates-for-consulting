import yaml
from pathlib import Path
from configs.schema import Config

def load_config(path: str = "params.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(**raw)
