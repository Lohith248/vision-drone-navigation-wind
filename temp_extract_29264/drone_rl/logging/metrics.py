"""
Metrics tracking and aggregation utilities.
"""
from collections import deque
from typing import Dict, Optional
import numpy as np


class MetricsTracker:
    """
    Tracks running averages of training metrics over a window.

    Parameters
    ----------
    window_size : int
        Number of recent values to keep for averaging.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._buffers: Dict[str, deque] = {}

    def update(self, key: str, value: float) -> None:
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.window_size)
        self._buffers[key].append(value)

    def get_mean(self, key: str) -> float:
        if key not in self._buffers or len(self._buffers[key]) == 0:
            return 0.0
        return float(np.mean(self._buffers[key]))

    def get_std(self, key: str) -> float:
        if key not in self._buffers or len(self._buffers[key]) == 0:
            return 0.0
        return float(np.std(self._buffers[key]))

    def get_latest(self, key: str) -> Optional[float]:
        if key not in self._buffers or len(self._buffers[key]) == 0:
            return None
        return self._buffers[key][-1]

    def summary(self) -> Dict[str, float]:
        return {f"{k}_mean": self.get_mean(k) for k in self._buffers}

    def compute_smoothness(self, actions: np.ndarray) -> float:
        """Compute action smoothness metric: mean ||a_t - a_{t-1}||²."""
        if len(actions) < 2:
            return 0.0
        diffs = np.diff(actions, axis=0)
        return float(np.mean(np.sum(diffs**2, axis=-1)))
