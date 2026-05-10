"""
On-policy rollout buffer for PPO with GAE computation and minibatch generation.
"""
from typing import Dict, Generator, Optional
import numpy as np
import torch


class RolloutBuffer:
    """
    Stores on-policy trajectories and computes GAE advantages.
    Pre-allocated numpy arrays, reused across rollouts.

    Parameters
    ----------
    n_steps : int
        Number of steps per rollout.
    n_envs : int
        Number of parallel environments.
    obs_dim : int
        Observation dimension.
    action_dim : int
        Action dimension.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: Optional[int] = None,
        action_dim: int = 3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        obs_specs: Optional[Dict[str, Dict]] = None,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.is_dict_obs = obs_specs is not None

        # Pre-allocate buffers
        if self.is_dict_obs:
            self.observations = {}
            for key, spec in obs_specs.items():
                shape = tuple(spec["shape"])
                dtype = spec.get("dtype", np.float32)
                self.observations[key] = np.zeros((n_steps, n_envs, *shape), dtype=dtype)
        else:
            if obs_dim is None:
                raise ValueError("obs_dim is required for non-dict observations")
            self.observations = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.bool_)

        # Computed after rollout
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action: np.ndarray, reward: np.ndarray,
            value: np.ndarray, log_prob: np.ndarray, done: np.ndarray) -> None:
        """Add one timestep of data from all envs."""
        if self.is_dict_obs:
            for key in self.observations:
                self.observations[key][self.ptr] = obs[key]
        else:
            self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        """
        Compute GAE-λ advantages and discounted returns.

        Parameters
        ----------
        last_values : (n_envs,) final value estimates.
        last_dones : (n_envs,) whether final state was terminal.
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
            else:
                next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step + 1].astype(np.float32)

            delta = (self.rewards[step] +
                     self.gamma * next_values * next_non_terminal -
                     self.values[step])
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int,
                        normalize_advantages: bool = True) -> Generator:
        """
        Yield shuffled minibatches of flattened rollout data as torch tensors.

        Parameters
        ----------
        batch_size : int
        normalize_advantages : bool
        """
        n_samples = self.n_steps * self.n_envs
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # Flatten (steps, envs, ...) -> (steps*envs, ...)
        if self.is_dict_obs:
            flat_obs = {
                key: value.reshape((n_samples, *value.shape[2:]))
                for key, value in self.observations.items()
            }
        else:
            flat_obs = self.observations.reshape(-1, self.obs_dim)
        flat_actions = self.actions.reshape(-1, self.action_dim)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_values = self.values.reshape(-1)

        if normalize_advantages:
            adv_mean = flat_advantages.mean()
            adv_std = flat_advantages.std() + 1e-8
            flat_advantages = (flat_advantages - adv_mean) / adv_std

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            if self.is_dict_obs:
                obs_batch = {
                    key: torch.from_numpy(value[batch_idx])
                    for key, value in flat_obs.items()
                }
            else:
                obs_batch = torch.from_numpy(flat_obs[batch_idx])

            yield {
                "observations": obs_batch,
                "actions": torch.from_numpy(flat_actions[batch_idx]),
                "log_probs": torch.from_numpy(flat_log_probs[batch_idx]),
                "advantages": torch.from_numpy(flat_advantages[batch_idx]),
                "returns": torch.from_numpy(flat_returns[batch_idx]),
                "values": torch.from_numpy(flat_values[batch_idx]),
            }

    def reset(self) -> None:
        """Reset buffer for next rollout."""
        self.ptr = 0
        self.full = False
