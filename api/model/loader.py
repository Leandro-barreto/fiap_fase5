"""
loader.py – utilities to load ML models
--------------------------------------

The ``load_model`` function looks up and loads a model file based
on the ticker symbol.  In the original project this was a Keras
model saved as ``models/lstm_model_<TICKER>.h5``【777383667817022†L4-L6】.

Here we provide a skeleton implementation that can be adapted to the
format of your attached model file.  You might use TensorFlow,
scikit‑learn, or another library to load your model.  Replace the
placeholder code with the appropriate loading logic for your project.
"""

import os
from typing import Any

import joblib
from tensorflow.keras.models import load_model as keras_load_model  # Optional dependency


def load_model(ticker: str) -> Any:
    """Load a pre‑trained ML model for the given ticker.

    This implementation attempts to load a Keras model saved under the
    ``models`` directory following the naming convention
    ``lstm_model_{ticker}.h5``.  If you are using a different model
    format or file naming scheme, adjust the path and loading logic
    accordingly.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol that identifies which model to load.

    Returns
    -------
    Any
        A loaded model object ready for inference.

    Raises
    ------
    FileNotFoundError
        If the expected model file cannot be found.
    """
    # Derive the expected filename
    filename = f"lstm_model_{ticker}.h5"
    model_path = os.path.join("models", filename)

    # Fallback to look for a joblib model if the Keras model does not exist
    joblib_path = os.path.join("models", f"{ticker}.pkl")

    if os.path.exists(model_path):
        return keras_load_model(model_path)
    elif os.path.exists(joblib_path):
        return joblib.load(joblib_path)
    else:
        raise FileNotFoundError(f"No model file found for ticker '{ticker}' at {model_path} or {joblib_path}")
