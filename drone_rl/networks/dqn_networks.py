"""
Dueling DQN network architecture.
"""
from typing import Tuple
import torch
import torch.nn as nn
from drone_rl.networks.feature_extractors import MLPExtractor, _init_weights


class DuelingDQN(nn.Module):
    """
    Dueling DQN: separate value and advantage streams.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,.))

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    n_actions : int
        Number of discrete actions.
    hidden_dims : tuple of int
        Hidden layer sizes.
    """

    def __init__(self, obs_dim: int, n_actions: int = 11,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.features = MLPExtractor(obs_dim, hidden_dims, activation="relu")
        feat_dim = self.features.output_dim

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, n_actions)
        )
        self.value_stream.apply(lambda m: _init_weights(m))
        self.advantage_stream.apply(lambda m: _init_weights(m))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns Q-values for all actions: (B, n_actions).
        """
        features = self.features(obs)
        value = self.value_stream(features)           # (B, 1)
        advantage = self.advantage_stream(features)    # (B, n_actions)
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values

    def q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value for specific actions. action: (B,) int64."""
        q_all = self.forward(obs)
        return q_all.gather(1, action.unsqueeze(-1).long()).squeeze(-1)
