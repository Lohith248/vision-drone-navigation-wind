"""
SAC networks: Gaussian actor with reparameterisation and twin Q-networks.
"""
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from drone_rl.networks.feature_extractors import MLPExtractor, _init_weights

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACActor(nn.Module):
    """
    Stochastic Gaussian actor for SAC with reparameterisation trick.
    Outputs tanh-squashed actions in [-1, 1].

    Parameters
    ----------
    obs_dim : int
    action_dim : int
    hidden_dims : tuple of int
    """

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.features = MLPExtractor(obs_dim, hidden_dims, activation="relu")
        feat_dim = self.features.output_dim
        self.mean_head = nn.Linear(feat_dim, action_dim)
        self.log_std_head = nn.Linear(feat_dim, action_dim)
        _init_weights(self.mean_head, gain=0.01)
        _init_weights(self.log_std_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action using reparameterisation trick.

        Returns
        -------
        action : (B, action_dim) in [-1, 1]
        log_prob : (B,)
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        # Reparameterisation trick
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (tanh of mean)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class SACQNetwork(nn.Module):
    """Single Q-network: Q(s, a) -> scalar."""

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.net.apply(lambda m: _init_weights(m))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class SACTwinQ(nn.Module):
    """Twin Q-networks for clipped double-Q learning."""

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q1 = SACQNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = SACQNetwork(obs_dim, action_dim, hidden_dims)

    def forward(self, obs: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
