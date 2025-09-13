"""Feature engineering utilities for the hiring prediction project.

This module defines helper functions that operate on the flattened
dataset returned by :func:`prepare_data.build_dataset`.  It
provides a function to split the input into model features and an
optional function to construct a ``ColumnTransformer`` that applies
scaling to numeric features and one‑hot encoding to categorical
features.  The implementation is inspired by the training pipeline
earlier created in ``training_pipeline.py``【86947837380131†L140-L239】.
"""

from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .prepare_data import build_dataset

def load_features(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Load the raw JSON files and return X, y and metadata.

    This is a thin wrapper around :func:`prepare_data.build_dataset`
    that simply forwards the ``data_dir`` argument.  It is defined
    here to provide a consistent entry point for feature engineering.

    Parameters
    ----------
    data_dir : Path
        Directory containing the raw JSON files.

    Returns
    -------
    tuple
        ``X`` (DataFrame with id, numeric and categorical columns),
        ``y`` (Series with binary labels) and ``meta`` (dict with
        column lists).
    """
    return build_dataset(data_dir)

def split_features(X: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Remove ID columns from the feature set.

    Many models should not use identifier columns for training.  This
    helper strips those out using the ``id_cols`` list in ``meta``.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of all features including id columns.
    meta : Dict
        Metadata dictionary returned by :func:`build_dataset` containing
        ``id_cols``.

    Returns
    -------
    pd.DataFrame
        DataFrame of features without ID columns.
    """
    id_cols = meta.get("id_cols", [])
    return X.drop(columns=id_cols, errors="ignore")

def get_preprocessor(meta: Dict) -> ColumnTransformer:
    """Construct a preprocessing transformer for numeric and categorical features.

    Numeric features are standardized using ``StandardScaler`` and
    categorical features are one‑hot encoded.  Unknown categories are
    ignored to allow the model to handle unseen values during inference.

    Parameters
    ----------
    meta : Dict
        Metadata dictionary containing lists of numeric and categorical
        column names under keys ``num_cols`` and ``cat_cols``.

    Returns
    -------
    ColumnTransformer
        A transformer suitable for use in a scikit‑learn Pipeline.
    """
    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor