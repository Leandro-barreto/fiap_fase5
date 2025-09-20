"""Tests for metric helper functions."""

import numpy as np

from src.utils import metrics


def test_compute_accuracy() -> None:
    """compute_accuracy should return the proportion of correct predictions."""
    y_true = [1, 0, 1, 1]
    y_pred = [1, 1, 1, 0]
    acc = metrics.compute_accuracy(y_true, y_pred)
    # Two of four predictions are correct
    assert acc == 0.5


def test_compute_f1() -> None:
    """compute_f1 should compute the F1 score for binary classification."""
    y_true = [1, 0, 1, 1]
    y_pred = [1, 1, 1, 0]
    f1 = metrics.compute_f1(y_true, y_pred)
    # Precision = 2/3, Recall = 2/3, F1 = 2/3
    assert np.isclose(f1, 2 / 3)