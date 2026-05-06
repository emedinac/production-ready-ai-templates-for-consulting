"""Data preprocessing module."""

from pathlib import Path
import json
import yaml
import pandas as pd
from datasets import load_dataset

from ...configs.loader import load_config


def preprocess() -> Path:
    config = load_config()

    # ---- COMPUTE PROJECT ROOT ----
    project_root = Path(__file__).resolve().parents[2]

    # ---- DVC-SAFE OUTPUT PATHS (hardcoded, not config-driven) ----
    output_dir = project_root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = project_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---- LOAD DATA ----
    if config.data.source.type == "huggingface":
        dataset = load_dataset(
            config.data.source.huggingface.name, config.data.source.huggingface.version
        )
        # Assume single split for simplicity, or take 'train' split
        df = pd.DataFrame(dataset["train"])
    elif config.data.source.type == "local":
        # Assume CSV for now
        df = pd.read_csv(config.data.source.local.path)
    else:
        raise NotImplementedError(
            f"Source type {config.data.source.type} not implemented"
        )

    # ---- APPLY PREPROCESSING ----
    if config.data.preprocessing:
        if config.task.type == "text_classification" and config.data.preprocessing.text:
            if config.data.preprocessing.text.lowercase:
                df["text"] = df["text"].str.lower()
            # Add other text preprocessing
        # Add tabular preprocessing if needed

    # ---- SPLIT DATA ----
    total_samples = len(df)
    train_end = int(total_samples * config.data.split.train)
    val_end = train_end + int(total_samples * config.data.split.validation)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # ---- SAVE SPLITS ----
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # ---- DATA ARTIFACT ----
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

    # ---- OPTIONAL: REAL METRICS ONLY IF COMPUTED ----
    metrics = {
        "status": "completed",
        "total_samples": total_samples,
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "num_files": len(list(output_dir.glob("*"))),
    }

    (metrics_dir / "preprocess.json").write_text(json.dumps(metrics, indent=2))

    return output_dir


if __name__ == "__main__":
    preprocess()
