"""Inference utilities for the hiring prediction model.

This module exposes functions to load a previously trained model
(including its preprocessing pipeline) and to perform predictions on
new data.  Because the model pipeline contains both the
preprocessing and the classifier, input data should have the same
feature columns used during training (excluding identifier columns).

Example
-------
>>> from pathlib import Path
>>> from src.models.infer import load_pipeline, predict
>>> model = load_pipeline(Path("./models/latest/model.joblib"))
>>> import pandas as pd
>>> X_new = pd.DataFrame({...})  # columns must match training
>>> probs = predict(model, X_new)

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import joblib
import numpy as np
import pandas as pd

def load_pipeline(model_path: Union[str, Path]):
    """Load a trained pipeline from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to the ``joblib`` file containing the trained pipeline.

    Returns
    -------
    Any
        The loaded scikit‑learn pipeline.
    """
    return joblib.load(model_path)

def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict labels for a batch of samples using the loaded model.

    Parameters
    ----------
    model : Any
        Trained scikit‑learn pipeline containing preprocessing and classifier.
    X : pd.DataFrame
        DataFrame of features (excluding id columns).  Must contain the
        same numeric and categorical columns used in training.

    Returns
    -------
    np.ndarray
        Array of predicted labels (0 or 1).
    """
    return model.predict(X)

def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return prediction probabilities for a batch of samples.

    This function returns the probability of the positive class (1).

    Parameters
    ----------
    model : Any
        Trained scikit‑learn pipeline.
    X : pd.DataFrame
        DataFrame of features matching the training schema.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) with probabilities of the positive class.
    """
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 else proba