"""
DQN algorithm: Double DQN + Dueling architecture + PER.
"""
import copy
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


class DQNAlgorithm:
    """
    Double Dueling DQN with Prioritized Experience Replay.

    Parameters
    ----------
    q_network : DuelingDQN
    n_actions : int
    lr : float
    gamma : float
    target_update_freq : int
    max_grad_norm : float
    use_amp : bool
    device : torch.device
    """

    def __init__(self, q_network, n_actions: int = 11, lr: float = 1e-4,
                 gamma: float = 0.99, target_update_freq: int = 500,
                 max_grad_norm: float = 10.0, use_amp: bool = True,
                 device: torch.device = torch.device("cuda")):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device.type == "cuda"

        self.q_network = q_network.to(device)
        self.target_network = copy.deepcopy(q_network).to(device)
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=1e-5)
        self.scaler = GradScaler() if self.use_amp else None
        self._update_count = 0

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_t)
            return int(q_values.argmax(dim=1).item())

    def update(self, batch: Dict[str, torch.Tensor],
               weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform one Double DQN update.

        Parameters
        ----------
        batch : dict with obs, actions, rewards, next_obs, dones
        weights : importance sampling weights from PER (optional)

        Returns
        -------
        metrics : dict
        td_errors : np.ndarray for PER priority update
        """
        obs = batch["observations"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]

        # Double DQN: select action with online, evaluate with target
        with torch.no_grad():
            # Online network selects best action
            next_q_online = self.q_network(next_obs)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            # Target network evaluates the selected action
            next_q_target = self.target_network(next_obs)
            next_q = next_q_target.gather(1, next_actions).squeeze(-1)
            td_target = rewards + self.gamma * (1 - dones) * next_q

        # Current Q-values
        current_q = self.q_network(obs)
        if actions.dim() == 2:
            q_pred = current_q.gather(1, actions).squeeze(-1)
        else:
            q_pred = current_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # TD errors for PER
        with torch.no_grad():
            td_errors = (q_pred - td_target).abs().cpu().numpy()

        # Loss (Huber / Smooth L1)
        if weights is not None:
            weights_t = torch.from_numpy(weights).float().to(self.device)
            element_loss = F.smooth_l1_loss(q_pred, td_target, reduction="none")
            loss = (element_loss * weights_t).mean()
        else:
            loss = F.smooth_l1_loss(q_pred, td_target)

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self._update_count += 1

        # Hard target update
        if self._update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        metrics = {
            "dqn_loss": loss.item(),
            "q_mean": q_pred.mean().item(),
            "q_max": q_pred.max().item(),
            "td_error_mean": td_errors.mean(),
        }
        return metrics, td_errors

    def state_dict(self) -> dict:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict):
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._update_count = state.get("update_count", 0)
