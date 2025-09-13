"""Metrics helpers for model evaluation.

This module wraps common scoring functions to provide a consistent
interface for computing evaluation metrics.  It currently supports
accuracy and F1 score, but you can add other metrics as needed.
"""

from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score

def compute_accuracy(y_true: Iterable, y_pred: Iterable) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : iterable
        True labels.
    y_pred : iterable
        Predicted labels.

    Returns
    -------
    float
        Accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true: Iterable, y_pred: Iterable) -> float:
    """Compute the F1 score for binary classification.

    Parameters
    ----------
    y_true : iterable
        True labels.
    y_pred : iterable
        Predicted labels.

    Returns
    -------
    float
        F1 score.
    """
    return f1_score(y_true, y_pred)