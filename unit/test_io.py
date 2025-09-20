"""Tests for I/O utilities.

These tests cover the helper functions in ``src/utils/io.py`` that read and
write JSON files and save/load Joblib models.  A temporary directory is
used to avoid polluting the working tree.
"""

from pathlib import Path

import pytest

from src.utils import io


def test_read_write_json(tmp_path: Path) -> None:
    """write_json followed by read_json should produce the original data."""
    data = {"foo": 1, "bar": {"baz": [1, 2, 3]}}
    path = tmp_path / "test.json"
    io.write_json(data, path)
    assert path.exists()
    loaded = io.read_json(path)
    assert loaded == data


def test_save_load_model(tmp_path: Path) -> None:
    """save_model should persist an object that can be loaded with load_model."""
    model = {"weights": [1, 2, 3]}
    path = tmp_path / "model.joblib"
    io.save_model(model, path)
    # file should exist after saving
    assert path.exists()
    loaded = io.load_model(path)
    assert loaded == model