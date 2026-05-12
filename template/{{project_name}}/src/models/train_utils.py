import os
import time
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

import mlflow
from mlflow.exceptions import MlflowException
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
    LinearRegression,
)


def setup_mlflow(experiment_name: str):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    deadline = time.time() + 60
    while True:
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                mlflow.create_experiment(experiment_name)

            mlflow.set_experiment(experiment_name)
            return
        except MlflowException as exc:
            if time.time() >= deadline:
                raise
            print(f"MLflow unavailable at {tracking_uri}, retrying: {exc}")
            time.sleep(2)


class TransformerWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):

        self.tokenizer = AutoTokenizer.from_pretrained(
            context.artifacts["tokenizer_path"]
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            context.artifacts["model_path"]
        )
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> list[int]:
        texts = model_input["text"].tolist()
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            out = self.model(**enc)

        return out.logits.argmax(-1).cpu().numpy()


# MODEL BUNDLE
@dataclass
class ModelBundle:
    model: Any
    tokenizer: Optional[Any] = None
    type: str = "sklearn"


# DATASET
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# TRANSFORMER TRAINING
def train_transformer(bundle: ModelBundle, train_df, val_df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = bundle.model.to(device)
    tokenizer = bundle.tokenizer

    dataset = TextDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
    )

    model.train()

    for epoch in range(config.training.epochs):
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Transformer] epoch={epoch + 1}, loss={total_loss:.4f}")

    return model


# MODEL BUILDER
def build_model(config):
    model_type = config.model.type.lower()
    problem_type = config.task.problem_type

    # SKLEARN
    if model_type in {"sklearn", "linear", "xgboost"}:
        if problem_type == "regression":
            model = (
                SGDRegressor(
                    max_iter=int(config.training.epochs),
                    eta0=float(config.training.learning_rate),
                )
                if config.training.optimizer.lower() == "sgd"
                else LinearRegression()
            )

        else:
            model = (
                SGDClassifier(
                    max_iter=int(config.training.epochs),
                    eta0=float(config.training.learning_rate),
                    tol=1e-3,
                )
                if config.training.optimizer.lower() == "sgd"
                else LogisticRegression(max_iter=300)
            )

        return ModelBundle(model=model, type="sklearn")

    # TRANSFORMER
    if model_type == "transformer":
        model_name = config.model.transformer.name

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=getattr(config.model, "num_labels", 2),
        )

        return ModelBundle(
            model=model,
            tokenizer=tokenizer,
            type="transformer",
        )

    raise ValueError(f"Unsupported model type: {model_type}")
