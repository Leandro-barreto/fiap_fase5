"""Tests for the FastAPI prediction endpoints.

These tests cover the main HTTP endpoints exposed by the API defined in
``api.main``.  They ensure the health check returns a simple status
response, that valid JSON and CSV payloads produce predictions and
probabilities with the expected structure, and that error conditions
such as missing models or invalid payloads yield appropriate HTTP
responses.

The module mirrors ``unit/api/test_api_routes.py`` but lives in a
different package (``api_2``) to avoid test collection issues on some
platforms.  The behaviour tested remains identical.
"""

import io
from typing import Dict

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")  # skip API tests if FastAPI is unavailable
from fastapi.testclient import TestClient  # type: ignore

from api.main import create_app


def test_health_endpoint() -> None:
    """GET /health should return a JSON object with status ok."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_candidate_json(monkeypatch, dummy_model) -> None:
    """POST /api/predict/candidate with JSON should return prediction and features."""
    # Patch model loader to return our dummy model
    monkeypatch.setattr("api.model.loader.load_model", lambda: dummy_model)
    app = create_app()
    client = TestClient(app)
    payload: Dict[str, str] = {
        "cv": "Conheço Python e pandas",
        "job_title": "Engenheiro de Dados",
        "job_description": "Conhecimentos em Python e SQL",
        "nivel_ingles": "Básico",
        "nivel_academico": "Graduação",
        "remuneracao": "3000",
        "tipo_contratacao": "CLT",
        "estado": "SP",
        "cidade": "São Paulo",
        "recrutador": "Joana",
        "analista_responsavel": "Carlos",
    }
    resp = client.post("/api/predict/candidate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # Should contain prediction, probability and features
    assert set(body.keys()) == {"prediction", "probability", "features"}
    assert isinstance(body["prediction"], int)
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["features"], dict)
    # Dummy model: overlap_kw must be computed; should exist in features
    assert "overlap_kw" in body["features"]


def test_predict_candidate_invalid_json(monkeypatch) -> None:
    """Sending non‑object JSON should result in a 400 error."""
    monkeypatch.setattr("api.model.loader.load_model", lambda: object())
    app = create_app()
    client = TestClient(app)
    # Send an array instead of a dict
    resp = client.post(
        "/api/predict/candidate",
        data="[1, 2, 3]",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


def test_predict_candidate_model_not_found(monkeypatch) -> None:
    """If the model cannot be loaded, the API should return 404."""

    def raise_fn():
        raise FileNotFoundError("not found")

    # Patch the load_model reference used inside the route.  The prediction
    # route imports load_model into its module namespace, so we must patch
    # ``api.routes.predict.load_model`` rather than ``api.model.loader.load_model``.
    monkeypatch.setattr("api.routes.predict.load_model", raise_fn)
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/predict/candidate", json={})
    assert resp.status_code == 404


def test_predict_candidate_csv(monkeypatch, dummy_model) -> None:
    """Uploading a CSV file should return lists of predictions and probabilities."""
    # Patch the load_model reference used by the route to return our dummy model
    monkeypatch.setattr("api.routes.predict.load_model", lambda: dummy_model)
    app = create_app()
    client = TestClient(app)
    # Create a DataFrame with required columns for dummy model
    df = pd.DataFrame({
        "sim_tfidf": [0.2, 0.8],
        "overlap_kw": [0.4, 0.6],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("features.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = client.post("/api/predict/candidate", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"predictions", "probabilities", "features"}
    # There should be one prediction per row
    assert len(body["predictions"]) == 2
    assert len(body["probabilities"]) == 2
    # The dummy model produces int predictions and float probabilities
    assert all(isinstance(p, int) for p in body["predictions"])
    assert all(isinstance(p, float) for p in body["probabilities"])
    # Features should be returned as a list of dicts
    assert isinstance(body["features"], list)
    assert len(body["features"]) == 2
