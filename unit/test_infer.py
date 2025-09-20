"""Tests for inference utilities.

These tests verify that functions in ``src/models/infer.py`` correctly
delegate to joblib and to the underlying model's methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import infer


def test_load_pipeline_delegates_to_joblib(monkeypatch) -> None:
    """load_pipeline should call joblib.load with the given path."""
    calls = {}

    def fake_load(path):
        calls["path"] = Path(path)
        return "model"

    # Replace the joblib module on infer with a dummy object having a load function
    fake_joblib = type("FakeJoblib", (), {"load": staticmethod(fake_load)})
    monkeypatch.setattr(infer, "joblib", fake_joblib)
    model = infer.load_pipeline("/tmp/model.joblib")
    assert model == "model"
    assert calls["path"] == Path("/tmp/model.joblib")


def test_predict_and_predict_proba_forward_calls() -> None:
    """predict and predict_proba should delegate to the model's methods."""
    class Dummy:
        def __init__(self):
            self.pred_called = False
            self.proba_called = False

        def predict(self, X):
            self.pred_called = True
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            self.proba_called = True
            # return uniform probabilities
            return np.column_stack((np.full(len(X), 0.3), np.full(len(X), 0.7)))

    dummy = Dummy()
    X = pd.DataFrame({"a": [1, 2, 3]})
    # Test predict
    preds = infer.predict(dummy, X)
    assert dummy.pred_called
    assert (preds == np.array([0, 0, 0])).all()
    # Test predict_proba
    probs = infer.predict_proba(dummy, X)
    assert dummy.proba_called
    # Should return only positive class probabilities
    assert np.allclose(probs, np.array([0.7, 0.7, 0.7]))