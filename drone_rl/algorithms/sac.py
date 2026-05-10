"""
SAC algorithm: twin Q, automatic entropy tuning, soft target updates.
"""
import copy
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


class SACAlgorithm:
    """
    Soft Actor-Critic with automatic entropy coefficient tuning.

    Parameters
    ----------
    actor : SACActor network
    critic : SACTwinQ network
    obs_dim : int
    action_dim : int
    lr_actor : float
    lr_critic : float
    lr_alpha : float
    gamma : float
    tau : float
    target_entropy : float or None (auto-computed)
    use_amp : bool
    device : torch.device
    """

    def __init__(self, actor, critic, obs_dim: int, action_dim: int = 3,
                 lr_actor: float = 3e-4, lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, target_entropy: float = None,
                 use_amp: bool = True,
                 device: torch.device = torch.device("cuda")):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.use_amp = use_amp and device.type == "cuda"

        # Networks
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device)
        # Freeze target
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

        # Auto entropy tuning
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=device,
                                      requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.scaler = GradScaler() if self.use_amp else None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one SAC update step.

        Parameters
        ----------
        batch : dict with keys observations, actions, rewards, next_observations, dones

        Returns
        -------
        metrics : dict
        """
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha.detach() * next_log_probs
            td_target = rewards + self.gamma * (1 - dones) * q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.scaler.step(self.critic_optimizer)
        else:
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.scaler.step(self.actor_optimizer)
        else:
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if self.use_amp:
            self.scaler.update()

        # --- Soft target update ---
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": ((q1.mean() + q2.mean()) / 2).item(),
            "log_prob_mean": log_probs.mean().item(),
        }

    def _soft_update(self):
        """Polyak averaging for target networks."""
        for p, p_target in zip(self.critic.parameters(),
                               self.critic_target.parameters()):
            p_target.data.mul_(1 - self.tau)
            p_target.data.add_(self.tau * p.data)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for a single observation."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
            return action.cpu().numpy().squeeze(0)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha = state["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_optimizer.defaults["lr"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
