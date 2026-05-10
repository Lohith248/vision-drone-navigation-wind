"""
DDPG algorithm: deterministic policy gradients with target networks.
"""
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class DDPGAlgorithm:
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
        self,
        actor,
        critic,
        action_dim: int = 3,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device.type == "cuda"

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.scaler = GradScaler() if self.use_amp else None

    def select_action(self, obs: np.ndarray, noise_std: float = 0.1,
                      deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.actor(obs_t).cpu().numpy().squeeze(0)
        if deterministic:
            return action
        noise = np.random.normal(0.0, noise_std, size=self.action_dim).astype(np.float32)
        return np.clip(action + noise, -1.0, 1.0)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_actions)
            td_target = rewards + self.gamma * (1.0 - dones) * target_q

        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            with autocast():
                current_q = self.critic(obs, actions)
                critic_loss = F.mse_loss(current_q, td_target)
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic_optimizer)
        else:
            current_q = self.critic(obs, actions)
            critic_loss = F.mse_loss(current_q, td_target)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # Actor update
        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            with autocast():
                pred_actions = self.actor(obs)
                actor_loss = -self.critic(obs, pred_actions).mean()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            pred_actions = self.actor(obs)
            actor_loss = -self.critic(obs, pred_actions).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        self._soft_update()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_mean": float(current_q.mean().item()),
        }

    def _soft_update(self) -> None:
        for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
