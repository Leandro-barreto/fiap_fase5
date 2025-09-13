"""Utility functions for I/O operations.

This module contains helper functions for reading and writing data
files commonly used in the project.  Having a central location for
I/O makes it easier to modify file handling (e.g. adding logging or
error handling) without touching multiple modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import joblib

def read_json(path: Path) -> Dict:
    """Read a JSON file into a Python dict.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: Dict, path: Path) -> None:
    """Write a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        Data to serialize.
    path : Path
        Destination path for the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_model(model: Any, path: Path) -> None:
    """Serialize a scikit‑learn model or pipeline using joblib.

    Parameters
    ----------
    model : Any
        Model or pipeline to save.
    path : Path
        Destination path (.joblib).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: Path) -> Any:
    """Load a model or pipeline from a joblib file.

    Parameters
    ----------
    path : Path
        Path to the .joblib file.

    Returns
    -------
    Any
        The loaded model.
    """
    return joblib.load(path)