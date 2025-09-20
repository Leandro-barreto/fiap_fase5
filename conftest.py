"""Pytest configuration and fixtures for the hiring prediction project.

This ``conftest.py`` module defines fixtures used across the test suite.  The
fixtures provide deterministic behaviour (via seeding and frozen time),
temporary data directories populated with toy JSON files to exercise data
preparation functions, and dummy models for API tests.  Keeping common
fixtures here reduces boilerplate in individual test modules.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

from src.utils.seed import set_seed

# -----------------------------------------------------------------------------
# Ensure project root is on sys.path
#
# When running tests from the ``unit`` package, the root directory of the
# project may not be present on Python's module search path.  This prevents
# imports like ``import api.main`` from resolving correctly.  The following
# block prepends the project root (the directory containing this file) to
# ``sys.path`` if it is not already present.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in map(str, sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def deterministic_seed() -> Iterator[None]:
    """Automatically set a deterministic seed for each test.

    This fixture runs before every test function (due to ``autouse=True``)
    and resets the global random generators to a fixed value.  Tests that
    depend on randomness should therefore produce deterministic results.
    """
    seed = 1234
    set_seed(seed)
    yield
    # Reset to a different seed after the test to avoid cross‑test leakage
    set_seed(seed + 1)


@pytest.fixture
def frozen_time() -> Iterator[None]:
    """Freeze time during tests that rely on the current date.

    Some functions may depend on the system clock.  Use this fixture to
    freeze time at the beginning of 2024 so tests remain reproducible.
    """
    with freeze_time("2024-01-01 00:00:00"):
        yield


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory containing toy JSON data files.

    The returned directory includes ``applicants.json``, ``prospects.json`` and
    ``vagas.json`` with small, consistent records.  This dataset is used for
    tests that exercise the full data preparation pipeline without
    performing heavy text vectorisation (by leaving free text fields empty).
    """
    data_dir = tmp_path / "data"
    # applicants: two simple candidates with minimal nested structure
    applicants = {
        "1001": {
            "infos_basicas": {
                "nome": "Alice",
                "nivel_ingles": "Intermediário",
                "nivel_academico": "Bacharelado",
            },
            "informacoes_profissionais": {
                "conhecimentos_tecnicos": "python pandas",
                "remuneracao": "2000",
            },
            "cv_pt": "engenheira de dados",
        },
        "1002": {
            "infos_basicas": {
                "nome": "Bob",
                # intentionally omit english level to trigger imputation
                "nivel_academico": None,
            },
            "informacoes_profissionais": {
                "conhecimentos_tecnicos": "java spark",
                # missing remuneration triggers default
            },
            # missing CV text triggers empty candidate text
        },
    }
    prospects = {
        "v1": {
            "titulo": "Engenheiro de Dados",
            "modalidade": "CLT",
            "prospects": [
                {"codigo": "1001", "situacao_candidato": "Contratado"},
                {"codigo": "1002", "situacao_candidato": "Rejeitado"},
            ],
        }
    }
    # vagas contain metadata such as contract type and location
    vagas = {
        "v1": {
            "informacoes_basicas": {
                "titulo_vaga": "Engenheiro de Dados",
                "tipo_contratacao": "CLT",
                "analista_responsavel": "Maria",
            },
            "perfil_vaga": {
                "estado": "SP",
                "cidade": "São Paulo",
            },
        }
    }
    _write_json(data_dir / "applicants.json", applicants)
    _write_json(data_dir / "prospects.json", prospects)
    _write_json(data_dir / "vagas.json", vagas)
    return data_dir


@pytest.fixture
def empty_prospects_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with empty prospects to trigger errors.

    Only the ``prospects.json`` file is empty; applicants and vacancies
    contain minimal valid structures.  ``build_dataset`` should raise
    ``ValueError`` when prospects are empty.
    """
    data_dir = tmp_path / "data_empty"
    applicants = {"1001": {"infos_basicas": {"nome": "Test"}}}
    prospects = {}  # empty prospects
    vagas = {}
    _write_json(data_dir / "applicants.json", applicants)
    _write_json(data_dir / "prospects.json", prospects)
    _write_json(data_dir / "vagas.json", vagas)
    return data_dir


@pytest.fixture
def dummy_model() -> object:
    """Return a dummy scikit‑learn like model for API tests.

    The returned object implements ``predict`` and ``predict_proba`` methods
    that accept a pandas DataFrame and return fixed predictions and
    probabilities.  This avoids loading a real model from disk.
    """

    class DummyModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            # Predict 1 if overlap_kw >= 0.5 else 0
            if "overlap_kw" in X.columns:
                return np.where(X["overlap_kw"] >= 0.5, 1, 0)
            # fallback: all zeros
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            # Probability increases with sim_tfidf and overlap_kw
            base = X.get("sim_tfidf", pd.Series(0.0, index=X.index))
            overlap = X.get("overlap_kw", pd.Series(0.0, index=X.index))
            proba = 0.5 * base + 0.5 * overlap
            # clip to [0, 1]
            proba = proba.clip(0, 1).to_numpy()
            # return as 2D array expected by sklearn (n_samples, 2)
            # negative class probability = 1 - positive class
            return np.column_stack((1 - proba, proba))

    return DummyModel()