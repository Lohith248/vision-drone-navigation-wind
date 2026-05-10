"""
DDPG networks: deterministic actor and critic Q-network.
"""
from typing import Tuple
import torch
import torch.nn as nn

from drone_rl.networks.feature_extractors import MLPExtractor, _init_weights


class DDPGActor(nn.Module):
    """Deterministic actor: obs -> action in [-1, 1]."""

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.features = MLPExtractor(obs_dim, hidden_dims, activation="relu")
        self.head = nn.Linear(self.features.output_dim, action_dim)
        _init_weights(self.head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.features(obs)
        return torch.tanh(self.head(x))


class DDPGCritic(nn.Module):
    """Critic Q-network: Q(s, a) -> scalar."""

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        prev_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(lambda m: _init_weights(m))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)
