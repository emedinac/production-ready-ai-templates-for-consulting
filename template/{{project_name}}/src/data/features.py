"""Feature preparation for training."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_text_classification_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, config
):
    """Prepare text classification data using config-driven preprocessing.

    Args:
        train_df: Training DataFrame with 'text' and 'label' columns.
        val_df: Validation DataFrame with same columns.
        config: Configuration object containing preprocessing settings.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, vectorizer).

    Raises:
        ValueError: If required columns are missing.
    """
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(
            "Text classification datasets must include 'text' and 'label' columns"
        )

    text_cfg = config.data.preprocessing.text if config.data.preprocessing else None
    if text_cfg is not None and text_cfg.max_length:
        train_df["text"] = (
            train_df["text"].astype(str).str.slice(0, text_cfg.max_length)
        )
        val_df["text"] = val_df["text"].astype(str).str.slice(0, text_cfg.max_length)

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"].astype(str))
    X_val = vectorizer.transform(val_df["text"].astype(str))

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    return X_train, X_val, y_train, y_val, vectorizer


def prepare_tabular_data(train_df: pd.DataFrame, val_df: pd.DataFrame, config):
    """Prepare tabular data with config-driven preprocessing.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        config: Configuration object containing preprocessing settings.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, scaler or None).
    """
    if "target" in train_df.columns:
        target_col = "target"
    else:
        target_col = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_col])
    X_val = val_df.drop(columns=[target_col])
    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()

    tabular_cfg = (
        config.data.preprocessing.tabular if config.data.preprocessing else None
    )
    if tabular_cfg is not None and tabular_cfg.handle_missing:
        if tabular_cfg.handle_missing == "drop":
            X_train = X_train.dropna()
            X_val = X_val.dropna()
        elif tabular_cfg.handle_missing == "median":
            X_train = X_train.fillna(X_train.median())
            X_val = X_val.fillna(X_train.median())
        elif tabular_cfg.handle_missing == "mode":
            X_train = X_train.fillna(X_train.mode().iloc[0])
            X_val = X_val.fillna(X_train.mode().iloc[0])
        else:
            X_train = X_train.fillna(X_train.mean())
            X_val = X_val.fillna(X_train.mean())

    if X_train.select_dtypes(include="object").shape[1] > 0:
        X_train = pd.get_dummies(X_train)
        X_val = pd.get_dummies(X_val).reindex(columns=X_train.columns, fill_value=0)

    if tabular_cfg is not None and tabular_cfg.normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    else:
        scaler = None

    return X_train, X_val, y_train, y_val, scaler
