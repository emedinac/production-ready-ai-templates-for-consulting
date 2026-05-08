from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

from tempfile import TemporaryDirectory

from ...configs.loader import load_config
from ...evaluation.metrics import compute_metrics
from ...src.data.features import (
    prepare_text_classification_data,
    prepare_tabular_data,
)
from .train_utils import build_model, train_transformer, TransformerWrapper

from sklearn.metrics import confusion_matrix, classification_report


def _resolve_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def train():
    config = load_config()
    repo_root = Path(__file__).resolve().parents[3]

    processed_dir = _resolve_path(repo_root, config.data.processed_dir)

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")

    if train_df.empty or val_df.empty:
        raise ValueError("Empty dataset")

    # DATA
    if config.task.type == "text_classification":
        X_train, X_val, y_train, y_val, feature_transformer = (
            prepare_text_classification_data(train_df, val_df, config)
        )
    else:
        X_train, X_val, y_train, y_val, feature_transformer = prepare_tabular_data(
            train_df, val_df, config
        )

    # MODEL
    bundle = build_model(config)

    if bundle.type == "transformer":
        model = train_transformer(bundle, train_df, val_df, config)
    else:
        model = bundle.model
        model.fit(X_train, y_train)

    # EVAL
    y_pred = model.predict(X_val)
    metrics = compute_metrics(config, y_val, y_pred)

    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    # MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    exp = mlflow.get_experiment_by_name(config.experiment.name)
    if exp is None:
        mlflow.create_experiment(config.experiment.name)
    mlflow.set_experiment(config.experiment.name)
    with mlflow.start_run():
        mlflow.log_params({
            "model_type": config.model.type,
            "task": config.task.type,
            "optimizer": config.training.optimizer,
            "lr": config.training.learning_rate,
            "epochs": config.training.epochs,
        })

        mlflow.log_metrics(metrics)

        # SIGNATURE (shared)
        signature = infer_signature(X_val, y_pred)

        # SKLEARN
        if bundle.type == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
            )

        # TRANSFORMER
        elif bundle.type == "transformer":
            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                model_path = tmp / "model"
                tokenizer_path = tmp / "tokenizer"

                model.save_pretrained(model_path)
                if bundle.tokenizer is not None:
                    bundle.tokenizer.save_pretrained(tokenizer_path)

                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=TransformerWrapper(),
                    artifacts={
                        "model_path": str(model_path),
                        "tokenizer_path": str(tokenizer_path),
                    },
                    signature=signature,
                )

        else:
            raise ValueError(f"Unsupported model type: {bundle.type}")

        # REGISTER MODEL
        model_name = (
            getattr(config.tracking.mlflow, "registered_model_name", None)
            or config.experiment.name
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        try:
            mlflow.register_model(model_uri, model_name)
        except Exception as e:
            print(f"[WARN] registration failed: {e}")

    print("Training complete")
