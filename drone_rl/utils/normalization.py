"""
Running statistics and observation normalization using Welford's online algorithm.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch


class RunningMeanStd:
    """
    Computes running mean and variance using Welford's online algorithm.
    Numerically stable even with large sample counts.

    Parameters
    ----------
    shape : tuple of int
        Shape of each observation vector.
    epsilon : float
        Small constant to prevent division by zero.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # initialise with epsilon to avoid div-by-zero

    def update(self, batch: np.ndarray) -> None:
        """
        Update statistics from a batch of observations.

        Parameters
        ----------
        batch : np.ndarray, shape (B, *shape)
            Batch of observations.
        """
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.shape):
            batch = batch[np.newaxis]  # add batch dim

        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray,
                             batch_count: int) -> None:
        """Merge batch moments with running moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize x using running statistics."""
        return (x - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + self.epsilon
        )

    def state_dict(self) -> dict:
        """Return serializable state."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from serialized state."""
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


class ObservationNormalizer:
    """
    Normalizes vector observations using RunningMeanStd.
    Handles dict observation spaces by normalizing only numeric (state) keys.

    Parameters
    ----------
    obs_dim : int
        Dimension of the vector observation.
    clip : float
        Maximum absolute value after normalization.
    """

    def __init__(self, obs_dim: int, clip: float = 10.0):
        self.rms = RunningMeanStd(shape=(obs_dim,))
        self.clip = clip

    def update(self, obs: np.ndarray) -> None:
        """Update running statistics with new observations."""
        self.rms.update(obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize and clip observations."""
        normalized = self.rms.normalize(obs)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        """Return serializable state."""
        return self.rms.state_dict()

    def load_state_dict(self, state: dict) -> None:
        """Restore from serialized state."""
        self.rms.load_state_dict(state)


class RewardNormalizer:
    """
    Normalize rewards using a running estimate of return variance.

    Parameters
    ----------
    gamma : float
        Discount factor for return estimation.
    epsilon : float
        Small constant for numerical stability.
    clip : float
        Maximum absolute value after normalization.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8,
                 clip: float = 10.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.rms = RunningMeanStd(shape=())
        self._return_estimate = 0.0

    def normalize(self, reward: float, done: bool) -> float:
        """Normalize a single reward value."""
        self._return_estimate = reward + self.gamma * self._return_estimate * (1.0 - float(done))
        self.rms.update(np.array([self._return_estimate]))
        normalized = reward / np.sqrt(self.rms.var + self.epsilon)
        return float(np.clip(normalized, -self.clip, self.clip))

    def state_dict(self) -> dict:
        return {"rms": self.rms.state_dict(), "return_estimate": self._return_estimate}

    def load_state_dict(self, state: dict) -> None:
        self.rms.load_state_dict(state["rms"])
        self._return_estimate = state["return_estimate"]
