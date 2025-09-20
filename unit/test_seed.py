"""Tests for random seed utilities."""

import random
import numpy as np

from src.utils.seed import set_seed


def test_set_seed_determinism() -> None:
    """set_seed should make random and numpy generators deterministic."""
    # Set the same seed twice and compare sequences
    set_seed(42)
    values1 = [random.random() for _ in range(3)]
    values1_np = np.random.rand(3).tolist()
    set_seed(42)
    values2 = [random.random() for _ in range(3)]
    values2_np = np.random.rand(3).tolist()
    assert values1 == values2
    assert values1_np == values2_np