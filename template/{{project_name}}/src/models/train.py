from pathlib import Path
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

from mlflow.models.signature import infer_signature
from tempfile import TemporaryDirectory
from sklearn.pipeline import Pipeline

from ...configs.loader import load_config
from ...evaluation.metrics import compute_metrics
from ...src.data.features import (
    prepare_text_classification_data,
    prepare_tabular_data,
)
from .train_utils import (
    build_model,
    train_transformer,
    TransformerWrapper,
    setup_mlflow,
)

from sklearn.metrics import confusion_matrix, classification_report


def resolve_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def train():
    config = load_config()

    experiment_name = config.experiment.name or "default"
    setup_mlflow(experiment_name)

    repo_root = Path(__file__).resolve().parents[3]
    processed_dir = resolve_path(repo_root, config.data.processed_dir)

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")

    if train_df.empty or val_df.empty:
        raise ValueError("Empty dataset")

    # DATA
    if config.task.type == "text_classification":
        X_train, X_val, y_train, y_val, preprocessor = prepare_text_classification_data(
            train_df, val_df, config
        )
    else:
        X_train, X_val, y_train, y_val, preprocessor = prepare_tabular_data(
            train_df, val_df, config
        )

    # MODEL
    bundle = build_model(config)

    if bundle.type == "transformer":
        model = train_transformer(bundle, train_df, val_df, config)
    else:
        model = bundle.model
        model.fit(X_train, y_train)

    # EVALUATION
    y_pred = model.predict(X_val)

    metrics = compute_metrics(config, y_val, y_pred)

    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    artifact_path = "model"

    # MLflow RUN
    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": config.model.type,
            "task": config.task.type,
            "optimizer": config.training.optimizer,
            "lr": config.training.learning_rate,
            "epochs": config.training.epochs,
        })

        mlflow.log_metrics(metrics)

        if config.task.type == "text_classification":
            serving_input = pd.DataFrame({"text": val_df["text"].astype(str)})
        else:
            serving_input = val_df.drop(columns=["target"], errors="ignore")

        signature = infer_signature(serving_input, y_pred)

        if bundle.type == "sklearn":
            if config.task.type == "text_classification":
                serving_model = Pipeline([
                    ("vectorizer", preprocessor),
                    ("model", model),
                ])
            else:
                serving_model = model

            mlflow.sklearn.log_model(
                sk_model=serving_model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=serving_input.head(5),
            )

        # TRANSFORMER MODEL
        elif bundle.type == "transformer":
            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                model_path = tmp / "model"
                tokenizer_path = tmp / "tokenizer"

                model.save_pretrained(model_path)

                if bundle.tokenizer:
                    bundle.tokenizer.save_pretrained(tokenizer_path)

                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=TransformerWrapper(),
                    artifacts={
                        "model_path": str(model_path),
                        "tokenizer_path": str(tokenizer_path),
                    },
                    signature=signature,
                )
        else:
            raise ValueError(f"Unsupported model type: {bundle.type}")

    print(f"Training complete. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train()
