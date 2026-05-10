"""
Deterministic seeding for reproducible experiments.
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set seeds for all random number generators used in training.

    Parameters
    ----------
    seed : int
        Global seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic mode (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
