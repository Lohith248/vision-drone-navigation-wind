"""
PPO algorithm: clipped objective, GAE, AMP, gradient clipping, LR scheduling.
"""
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


class PPOAlgorithm:
    """
    Proximal Policy Optimization with clipped objective.

    Parameters
    ----------
    policy : ActorCritic network
    lr : float
    clip_range : float
    ent_coef : float
    vf_coef : float
    max_grad_norm : float
    n_epochs : int
    use_amp : bool
    clip_vloss : bool
    device : torch.device
    """

    def __init__(self, policy, lr: float = 3e-4, clip_range: float = 0.2,
                 ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_epochs: int = 10,
                 use_amp: bool = True, clip_vloss: bool = True,
                 normalize_advantages: bool = True,
                 device: torch.device = torch.device("cuda")):
        self.policy = policy.to(device)
        self.device = device
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.use_amp = use_amp and device.type == "cuda"
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.scaler = GradScaler() if self.use_amp else None

    def update(self, rollout_buffer, batch_size: int = 512,
               lr: float = None) -> Dict[str, float]:
        """
        Perform PPO update over collected rollouts.

        Parameters
        ----------
        rollout_buffer : RolloutBuffer with computed GAE.
        batch_size : int
        lr : float, optional — override learning rate.

        Returns
        -------
        metrics : dict
        """
        if lr is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        all_metrics = {"policy_loss": [], "value_loss": [], "entropy": [],
                       "clip_fraction": [], "approx_kl": [], "explained_var": []}

        for epoch in range(self.n_epochs):
            for batch in rollout_buffer.get_minibatches(
                batch_size, normalize_advantages=self.normalize_advantages
            ):
                obs = self._move_obs_to_device(batch["observations"])
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                old_values = batch["values"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with autocast():
                        metrics = self._compute_loss(
                            obs, actions, old_log_probs, advantages,
                            returns, old_values
                        )
                    self.scaler.scale(metrics["total_loss"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(),
                                             self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    metrics = self._compute_loss(
                        obs, actions, old_log_probs, advantages,
                        returns, old_values
                    )
                    metrics["total_loss"].backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(),
                                             self.max_grad_norm)
                    self.optimizer.step()

                for k in all_metrics:
                    if k in metrics:
                        all_metrics[k].append(metrics[k].item() if torch.is_tensor(metrics[k]) else metrics[k])

        return {k: float(np.mean(v)) for k, v in all_metrics.items() if v}

    def _compute_loss(self, obs, actions, old_log_probs, advantages,
                      returns, old_values):
        _, log_prob, entropy, value = self.policy.get_action_and_value(obs, actions)

        # Policy loss (clipped surrogate)
        ratio = torch.exp(log_prob - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range,
                            1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (optionally clipped)
        if self.clip_vloss:
            v_clipped = old_values + torch.clamp(
                value - old_values, -self.clip_range, self.clip_range
            )
            v_loss1 = F.mse_loss(value, returns)
            v_loss2 = F.mse_loss(v_clipped, returns)
            value_loss = torch.max(v_loss1, v_loss2)
        else:
            value_loss = F.mse_loss(value, returns)

        entropy_loss = -entropy.mean()
        total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        with torch.no_grad():
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            var_returns = torch.var(returns)
            explained_var = 1 - F.mse_loss(value, returns) / (var_returns + 1e-8)

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "explained_var": explained_var,
        }

    def _move_obs_to_device(self, obs):
        if isinstance(obs, dict):
            return {key: value.to(self.device) for key, value in obs.items()}
        return obs.to(self.device)
