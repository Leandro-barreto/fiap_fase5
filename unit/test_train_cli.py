"""Tests for the train module's command‑line interface."""

import sys
from pathlib import Path

import pytest

from src.models import train as train_module


def test_cli_invokes_train_model(monkeypatch, tmp_path: Path, capsys) -> None:
    """Running main() should parse CLI arguments and call train_model."""
    called = {}

    def fake_train_model(data_dir: Path, model_output=None, test_size=0.2, random_state=42):
        called["data_dir"] = Path(data_dir)
        called["model_output"] = model_output
        called["test_size"] = test_size
        called["random_state"] = random_state

    monkeypatch.setattr(train_module, "train_model", fake_train_model)
    # Prepare arguments
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_path = tmp_path / "out.joblib"
    args = [
        "train.py",
        "--data-dir",
        str(data_dir),
        "--model-output",
        str(out_path),
        "--test-size",
        "0.3",
        "--random-state",
        "7",
    ]
    monkeypatch.setattr(sys, "argv", args)
    # Invoke CLI
    train_module.main()
    # Assertions
    assert called["data_dir"] == data_dir
    assert called["model_output"] == out_path
    assert pytest.approx(called["test_size"], rel=1e-6) == 0.3
    assert called["random_state"] == 7