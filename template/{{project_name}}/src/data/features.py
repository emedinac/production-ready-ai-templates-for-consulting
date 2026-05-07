"""Feature preparation for training."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_text_classification_data(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Prepare text classification data with TF-IDF vectorization.

    Args:
        train_df: Training DataFrame with 'text' and 'label' columns.
        val_df: Validation DataFrame with same columns.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, vectorizer).

    Raises:
        ValueError: If required columns are missing.
    """
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(
            "Text classification datasets must include 'text' and 'label' columns"
        )

    vectorizer = TfidfVectorizer(max_features=10_000, max_df=0.95)
    X_train = vectorizer.fit_transform(train_df["text"].astype(str))
    X_val = vectorizer.transform(val_df["text"].astype(str))

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    return X_train, X_val, y_train, y_val, vectorizer


def prepare_tabular_data(train_df: pd.DataFrame, val_df: pd.DataFrame, normalize: bool):
    """Prepare tabular data with optional normalization.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        normalize: Whether to apply StandardScaler normalization.

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

    if X_train.select_dtypes(include="object").shape[1] > 0:
        X_train = pd.get_dummies(X_train)
        X_val = pd.get_dummies(X_val).reindex(columns=X_train.columns, fill_value=0)

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    else:
        scaler = None

    return X_train, X_val, y_train, y_val, scaler
