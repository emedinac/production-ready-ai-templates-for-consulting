"""Data preprocessing module."""

from pathlib import Path
import json
import yaml
import pandas as pd
from datasets import load_dataset

from ...configs.loader import load_config


def _resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _fill_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "drop":
        return df.dropna()

    numeric_columns = df.select_dtypes(include=["number"]).columns
    if strategy == "median":
        return df.fillna(df[numeric_columns].median())
    if strategy == "mode":
        return df.fillna(df[numeric_columns].mode().iloc[0])
    return df.fillna(df[numeric_columns].mean())


def _apply_text_preprocessing(df: pd.DataFrame, text_cfg):
    if text_cfg.lowercase:
        df["text"] = df["text"].astype(str).str.lower()
    if text_cfg.max_length:
        df["text"] = df["text"].astype(str).str.slice(0, text_cfg.max_length)
    return df


def preprocess() -> Path:
    config = load_config()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = _resolve_path(repo_root, config.data.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = _resolve_path(repo_root, config.artifacts.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    if config.data.source.type == "huggingface":
        dataset = load_dataset(
            config.data.source.huggingface.name,
            config.data.source.huggingface.version,
        )
        df = pd.DataFrame(dataset["train"])
    elif config.data.source.type == "local":
        data_path = _resolve_path(repo_root, config.data.source.local.path)
        df = pd.read_csv(data_path)
    else:
        raise NotImplementedError(
            f"Source type {config.data.source.type} not implemented"
        )

    # APPLY PREPROCESSING
    if config.data.preprocessing:
        if config.task.type == "text_classification" and config.data.preprocessing.text:
            df = _apply_text_preprocessing(df, config.data.preprocessing.text)
        if (
            config.task.type == "tabular_regression"
            and config.data.preprocessing.tabular
        ):
            df = _fill_missing(df, config.data.preprocessing.tabular.handle_missing)

    # SPLIT DATA
    total_samples = len(df)
    train_end = int(total_samples * config.data.split.train)
    val_end = train_end + int(total_samples * config.data.split.validation)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # SAVE SPLITS
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    metadata = {
        "task": config.task.type,
        "problem_type": config.task.problem_type,
        "source_type": config.data.source.type,
        "split": config.data.split.model_dump(),
        "total_samples": total_samples,
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
    }

    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=True))

    metrics = {
        "status": "completed",
        "total_samples": total_samples,
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "num_files": len(list(output_dir.glob("*"))),
    }

    (metrics_dir / "preprocess.json").write_text(json.dumps(metrics, indent=2))

    results_experiment_dir = repo_root / "results" / config.experiment.name
    results_metrics_dir = results_experiment_dir / "metrics"
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    (results_metrics_dir / "preprocess.json").write_text(json.dumps(metrics, indent=2))

    return output_dir


if __name__ == "__main__":
    preprocess()
