"""Random seed configuration utilities.

Setting seeds for all random number generators helps ensure that
results are reproducible across runs.  This module provides a
convenient function to set seeds for Python's ``random`` module,
NumPy and PyTorch if available.
"""

from __future__ import annotations

import random
import os

import numpy as np

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # torch may not be installed

def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy and optionally PyTorch.

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behaviour
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False