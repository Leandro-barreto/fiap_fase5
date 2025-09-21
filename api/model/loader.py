"""Model loader for the hiring prediction API using LightGBM.

This module defines a simple helper to load the trained machine
learning pipeline used by the API.  The pipeline is persisted as a
Joblib file in the ``models`` directory.  The default model file has
been updated to ``lgbm_model.joblib`` to reflect the switch from
previous estimators to a LightGBM classifier.  If the model file
cannot be found an informative ``FileNotFoundError`` is raised.
"""

import os
import joblib
from typing import Any


# Name of the persisted model file.  The training pipeline persists
# the LightGBM model to this filename.  When updating the model
# version simply change this constant.
MODEL_FILENAME = "lgbm_model.joblib"

# Directory containing trained models.  The API expects the model file
# to reside under this directory relative to the project root.
MODEL_DIR = "models"

# Full path to the model file.  This is resolved at runtime so that
# relative paths work regardless of the current working directory.
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def load_model(_: str = "") -> Any:
    """Load and return the hiring prediction pipeline.

    Parameters
    ----------
    _ : str, optional
        Unused parameter kept for backward compatibility.  The model
        loaded does not depend on an external identifier.

    Returns
    -------
    Any
        A scikit‑learn pipeline containing preprocessing and a
        LightGBM classifier.

    Raises
    ------
    FileNotFoundError
        If the expected model file cannot be found.
    """
    # Verify the model file exists before attempting to load it.  If
    # missing, provide a clear error message pointing to the expected
    # location.  This helps users diagnose configuration issues when
    # deploying the API.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    return joblib.load(MODEL_PATH)