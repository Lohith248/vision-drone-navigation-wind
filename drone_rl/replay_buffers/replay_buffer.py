"""
Uniform experience replay buffer for off-policy algorithms (SAC, DQN).
Pre-allocated circular buffer with efficient batch sampling.
"""
from typing import Dict
import numpy as np
import torch


class ReplayBuffer:
    """
    Circular replay buffer with pre-allocated numpy arrays.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_dim : int
        Observation vector dimension.
    action_dim : int
        Action dimension (3 for continuous, 1 for discrete).
    action_dtype : str
        "float32" for continuous, "int64" for discrete.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int = 3,
                 action_dtype: str = "float32"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        np_dtype = np.float32 if action_dtype == "float32" else np.int64

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np_dtype)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def push(self, obs: np.ndarray, action: np.ndarray, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        """Store a single transition."""
        idx = self._ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                   next_obs: np.ndarray, dones: np.ndarray) -> None:
        """Store a batch of transitions."""
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.push(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch and return as GPU tensors.

        Returns dict with keys: observations, actions, rewards, next_observations, dones
        """
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return {
            "observations": torch.from_numpy(self.observations[indices]).to(device),
            "actions": torch.from_numpy(self.actions[indices]).to(device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(device),
            "next_observations": torch.from_numpy(self.next_observations[indices]).to(device),
            "dones": torch.from_numpy(self.dones[indices]).to(device),
        }

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        obs_bytes = self.observations.nbytes + self.next_observations.nbytes
        action_bytes = self.actions.nbytes
        scalar_bytes = self.rewards.nbytes + self.dones.nbytes
        return (obs_bytes + action_bytes + scalar_bytes) / 1024**2
