"""
Prioritized Experience Replay (PER) with sum-tree for O(log n) sampling.
"""
from typing import Dict, Tuple
import numpy as np
import torch


class SumTree:
    """Binary sum-tree for efficient priority-based sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0

    def update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority: float) -> int:
        tree_idx = self.data_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        data_idx = self.data_ptr
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        return data_idx

    def get(self, value: float) -> Tuple[int, float, int]:
        """Retrieve leaf (tree_idx, priority, data_idx) for given cumulative value."""
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left] or right >= len(self.tree):
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_idx = parent - self.capacity + 1
        return parent, self.tree[parent], data_idx

    @property
    def total_priority(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        return np.max(self.tree[self.capacity - 1:self.capacity - 1 + self.capacity])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Parameters
    ----------
    capacity : int
    obs_dim : int
    action_dim : int
    alpha : float
        Priority exponent (0 = uniform, 1 = full prioritization).
    beta_start : float
        Initial importance sampling exponent.
    beta_end : float
        Final importance sampling exponent (annealed over training).
    action_dtype : str
        "float32" for continuous, "int64" for discrete.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int = 1,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0, action_dtype: str = "int64"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._beta = beta_start

        np_dtype = np.float32 if action_dtype == "float32" else np.int64

        self.tree = SumTree(capacity)
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np_dtype)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._max_priority = 1.0
        self._epsilon = 1e-6

    def push(self, obs: np.ndarray, action, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        priority = self._max_priority ** self.alpha
        data_idx = self.tree.add(priority)
        self.observations[data_idx] = obs
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_observations[data_idx] = next_obs
        self.dones[data_idx] = float(done)
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu"),
               progress: float = 0.0) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample a prioritized batch.

        Returns
        -------
        batch : dict of tensors
        tree_indices : np.ndarray for priority updates
        weights : np.ndarray of importance sampling weights
        """
        # Anneal beta
        self._beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total_priority / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            tree_idx, priority, data_idx = self.tree.get(value)
            tree_indices[i] = tree_idx
            indices[i] = data_idx
            priorities[i] = max(priority, self._epsilon)

        # Importance sampling weights
        probs = priorities / self.tree.total_priority
        weights = (self._size * probs) ** (-self._beta)
        weights = weights / weights.max()

        batch = {
            "observations": torch.from_numpy(self.observations[indices]).to(device),
            "actions": torch.from_numpy(self.actions[indices]).to(device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(device),
            "next_observations": torch.from_numpy(self.next_observations[indices]).to(device),
            "dones": torch.from_numpy(self.dones[indices]).to(device),
        }
        return batch, tree_indices, weights.astype(np.float32)

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, td in zip(tree_indices, td_errors):
            priority = (abs(td) + self._epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size
