"""Tests for the feature engineering utilities.

These tests verify that the helper functions in ``src/data/feature_engineering.py``
behave as expected.  ``split_features`` should drop identifier columns
according to the metadata, ``get_preprocessor`` should generate a
``ColumnTransformer`` that transforms numeric and categorical columns
appropriately, and ``load_features`` should delegate to ``build_dataset``.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import feature_engineering as fe


def test_split_features_drops_id_columns() -> None:
    """split_features should remove id columns specified in meta."""
    X = pd.DataFrame({
        "id": ["a", "b"],
        "num": [1.0, 2.0],
        "cat": ["x", "y"],
    })
    meta = {"id_cols": ["id"]}
    out = fe.split_features(X, meta)
    assert "id" not in out.columns
    assert list(out.columns) == ["num", "cat"]


def test_get_preprocessor_transforms_columns() -> None:
    """get_preprocessor should build a transformer that handles numeric and categorical data."""
    meta = {
        "num_cols": ["num"],
        "cat_cols": ["cat"],
    }
    preproc = fe.get_preprocessor(meta)
    assert isinstance(preproc, ColumnTransformer)
    # Fit on a tiny dataset
    X = pd.DataFrame({
        "num": [0.0, 1.0, 2.0],
        "cat": ["a", "b", "a"],
    })
    Xt = preproc.fit_transform(X)
    # After standardisation: mean 1, std sqrt(2)
    scaled = (X["num"] - X["num"].mean()) / X["num"].std()
    # One-hot encoding yields two columns
    assert Xt.shape[1] == 1 + 2
    # Check that the numeric transformer is a StandardScaler
    # After fitting the ColumnTransformer, named_transformers_ holds a mapping
    # from transformer names to fitted transformer pipelines
    transformers = preproc.named_transformers_
    num_pipeline = transformers["num"]
    cat_pipeline = transformers["cat"]
    # The first step of the numeric pipeline is a StandardScaler
    assert isinstance(num_pipeline[0], StandardScaler)
    # The first step of the categorical pipeline is a OneHotEncoder
    assert isinstance(cat_pipeline[0], OneHotEncoder)


def test_load_features_calls_build_dataset(monkeypatch, tmp_path: Path) -> None:
    """load_features should return whatever build_dataset returns."""
    # Prepare dummy return values
    X_dummy = pd.DataFrame({"a": [1, 2]})
    y_dummy = pd.Series([0, 1])
    meta_dummy: Dict = {"id_cols": [], "num_cols": [], "cat_cols": []}

    def fake_build_dataset(data_dir: Path):
        assert data_dir == Path("/some/dir")
        return X_dummy, y_dummy, meta_dummy

    monkeypatch.setattr(fe, "build_dataset", fake_build_dataset)
    out_X, out_y, out_meta = fe.load_features(Path("/some/dir"))
    assert out_X is X_dummy
    assert out_y is y_dummy
    assert out_meta is meta_dummy