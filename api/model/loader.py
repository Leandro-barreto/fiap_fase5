"""Model loader for the hiring prediction API.

The trained scikit‑learn pipeline is persisted as a Joblib file under
``models/contratacao_model.joblib``.  This module provides a helper
function to load that pipeline, ignoring any ticker or identifier.
"""

import os
import joblib
from typing import Any

MODEL_FILENAME = "contratacao_model.joblib"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def load_model(_: str = "") -> Any:
    """Load and return the hiring prediction pipeline.

    Parameters
    ----------
    _ : str, optional
        Unused parameter kept for backward compatibility.  The model
        loaded does not depend on a ticker or any external identifier.

    Returns
    -------
    Any
        A scikit‑learn pipeline containing preprocessing and classifier.

    Raises
    ------
    FileNotFoundError
        If the expected model file cannot be found.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    return joblib.load(MODEL_PATH)