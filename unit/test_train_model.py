"""Tests for the training pipeline.

The training function ``src/models/train.train_model`` orchestrates data
loading, preprocessing, oversampling, model fitting and optional SHAP
explanation.  These tests patch out expensive or side‑effectful components
such as SHAP and model persistence to keep runtime minimal.  They check
that the training logic runs end‑to‑end on a small dataset and that
arguments are passed through correctly.
"""

import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import train as train_module


def _create_small_dataset():
    """Return a small dataset for training tests.

    The dataset contains four examples with numeric and categorical
    features.  Labels are balanced to exercise oversampling.
    """
    X = pd.DataFrame({
        "id_col": ["a", "b", "c", "d"],
        "sim_tfidf": [0.1, 0.2, 0.3, 0.4],
        "overlap_kw": [0.0, 0.5, 0.0, 0.5],
        "remuneracao_num": [1000, 2000, 1500, 1800],
        "tempo_processamento": [10, 20, 5, 15],
        "cand_missing_ratio": [0.0, 0.0, 0.0, 0.0],
        "cand_text_len": [10, 20, 15, 25],
        "vaga_text_len": [12, 22, 18, 28],
        "nivel_academico": ["Bacharelado", "Mestrado", "Doutorado", "Mestrado"],
        "nivel_ingles": ["Intermediário", "Avançado", "Fluente", "Nenhum"],
        "tipo_contratacao": ["CLT", "PJ", "CLT", "CLT"],
        "estado": ["SP", "RJ", "SP", "MG"],
        "cidade": ["São Paulo", "Rio", "Campinas", "Belo Horizonte"],
        "recrutador": ["RecA", "RecB", "RecA", "RecC"],
        "analista_responsavel": ["AnaA", "AnaB", "AnaA", "AnaC"],
    })
    y = pd.Series([1, 0, 0, 1])
    meta = {
        "id_cols": ["id_col"],
        "num_cols": [
            "sim_tfidf",
            "overlap_kw",
            "remuneracao_num",
            "tempo_processamento",
            "cand_missing_ratio",
            "cand_text_len",
            "vaga_text_len",
        ],
        "cat_cols": [
            "nivel_academico",
            "nivel_ingles",
            "tipo_contratacao",
            "estado",
            "cidade",
            "recrutador",
            "analista_responsavel",
        ],
    }
    return X, y, meta


def test_train_model_runs(monkeypatch, tmp_path: Path) -> None:
    """train_model should execute end‑to‑end without SHAP or saving the model."""
    # Patch load_features to return our small dataset
    def fake_load_features(data_dir: Path):
        return _create_small_dataset()

    monkeypatch.setattr(train_module.fe, "load_features", fake_load_features)
    monkeypatch.setattr(train_module.fe, "split_features", lambda X, meta: X.drop(columns=meta["id_cols"]))
    monkeypatch.setattr(train_module.fe, "get_preprocessor", lambda meta: train_module.fe.get_preprocessor(meta))

    # Patch joblib.dump to a no‑op so nothing is written to disk
    calls = []

    def fake_dump(obj, path):
        calls.append(path)

    monkeypatch.setattr(train_module.joblib, "dump", fake_dump)

    # Patch RandomOverSampler to a no‑op to avoid errors on tiny datasets.  The
    # dummy oversampler simply returns the inputs unchanged.
    class DummyROS:
        def __init__(self, *args, **kwargs):
            pass

        def fit_resample(self, X, y):
            return X, y

    monkeypatch.setattr(train_module, "RandomOverSampler", DummyROS)

    # Ensure import of shap raises ImportError so the SHAP branch is skipped
    # We save a reference to the original __import__ so our fake_import can
    # delegate to it; otherwise recursion would occur when the fake calls itself.
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        if name == "shap":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Run training; should not raise
    train_module.train_model(Path("/dummy/data"), model_output=None, test_size=0.5, random_state=0)
    # Since model_output is None, joblib.dump should not be called
    assert calls == []


def test_train_model_saves_model(monkeypatch, tmp_path: Path) -> None:
    """When model_output is provided, train_model should persist the model."""
    # Reuse dataset fixture
    monkeypatch.setattr(train_module.fe, "load_features", lambda data_dir: _create_small_dataset())
    monkeypatch.setattr(train_module.fe, "split_features", lambda X, meta: X.drop(columns=meta["id_cols"]))
    monkeypatch.setattr(train_module.fe, "get_preprocessor", lambda meta: train_module.fe.get_preprocessor(meta))
    # Skip SHAP again using the original import.  We avoid recursion by
    # delegating to the real __import__ stored before patching.
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        if name == "shap":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Track saved path
    saved = {}

    def fake_dump(obj, path):
        saved["path"] = Path(path)

    monkeypatch.setattr(train_module.joblib, "dump", fake_dump)

    # Patch RandomOverSampler again for this test
    class DummyROS:
        def __init__(self, *args, **kwargs):
            pass

        def fit_resample(self, X, y):
            return X, y

    monkeypatch.setattr(train_module, "RandomOverSampler", DummyROS)

    model_output = tmp_path / "model.joblib"
    train_module.train_model(Path("/dummy/data"), model_output=model_output, test_size=0.5, random_state=0)
    # Check that joblib.dump was invoked with the provided path
    assert saved.get("path") == model_output