from dataclasses import dataclass
from typing import Any, Optional

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


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Optional[Any] = None
    type: str = "sklearn"


# SIMPLE TEXT DATASET
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


# TRANSFORMER TRAINING LOOP
def train_transformer(bundle: ModelBundle, train_df, val_df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = bundle.model.to(device)
    tokenizer = bundle.tokenizer

    train_dataset = TextDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
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

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Transformer] Epoch {epoch + 1}, loss={total_loss:.4f}")

    return model


# MODEL BUILDER
def build_model(config):
    optimizer = config.training.optimizer.lower()
    epochs = int(config.training.epochs)
    problem_type = config.task.problem_type
    model_type = config.model.type.lower()

    # SKLEARN
    if model_type in {"sklearn", "linear", "xgboost"}:
        if problem_type == "regression":
            model = (
                SGDRegressor(
                    max_iter=epochs,
                    eta0=float(config.training.learning_rate),
                    learning_rate="invscaling",
                )
                if optimizer == "sgd"
                else LinearRegression()
            )

        elif problem_type in {"binary", "multiclass"}:
            model = (
                SGDClassifier(
                    max_iter=epochs,
                    eta0=float(config.training.learning_rate),
                    learning_rate="optimal",
                    tol=1e-3,
                )
                if optimizer == "sgd"
                else LogisticRegression(max_iter=max(epochs, 200))
            )
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

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
