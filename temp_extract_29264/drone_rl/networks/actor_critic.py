"""
Actor-Critic network for PPO with Gaussian policy.
"""
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from drone_rl.networks.feature_extractors import (
    CNNExtractor,
    MLPExtractor,
    VisionTransformerExtractor,
    _init_weights,
)

Observation = Union[torch.Tensor, Dict[str, torch.Tensor]]


class ActorCritic(nn.Module):
    """
    Actor-Critic for continuous control (PPO).

    - Actor outputs mean of Gaussian; log_std is a learnable parameter
    - Critic outputs scalar value estimate
    - Actions output in [-1, 1] via tanh (env scales to [-3, 3])

    Parameters
    ----------
    obs_dim : int
        Observation vector dimension.
    action_dim : int
        Action dimension (3 for vx, vy, vz).
    hidden_dims : tuple of int
        Hidden layer sizes for feature extractor.
    log_std_init : float
        Initial value for log standard deviation.
    activation : str
        Activation function name.
    """

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_init: float = -0.5,
        activation: str = "tanh",
        multimodal: bool = False,
        state_dim: Optional[int] = None,
        state_hidden_dims: Tuple[int, ...] = (128,),
        image_encoder: str = "vit",
        image_feature_dim: int = 192,
        vit_patch_size: int = 8,
        vit_embed_dim: int = 128,
        vit_depth: int = 4,
        vit_heads: int = 4,
        vit_dropout: float = 0.0,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.multimodal = multimodal

        if self.multimodal:
            if state_dim is None:
                raise ValueError("state_dim is required for multimodal ActorCritic")

            self.obs_dim = state_dim
            self.state_encoder = MLPExtractor(state_dim, state_hidden_dims, activation)
            if image_encoder == "vit":
                self.image_encoder = VisionTransformerExtractor(
                    output_dim=image_feature_dim,
                    patch_size=vit_patch_size,
                    embed_dim=vit_embed_dim,
                    depth=vit_depth,
                    num_heads=vit_heads,
                    dropout=vit_dropout,
                )
            elif image_encoder == "cnn":
                self.image_encoder = CNNExtractor(output_dim=image_feature_dim)
            else:
                raise ValueError(f"Unknown image encoder: {image_encoder}")
            fusion_dim = self.state_encoder.output_dim + self.image_encoder.output_dim
            self.features = MLPExtractor(fusion_dim, hidden_dims, activation)
        else:
            if obs_dim is None:
                raise ValueError("obs_dim is required for state-only ActorCritic")
            self.obs_dim = obs_dim
            self.state_encoder = None
            self.image_encoder = None
            self.features = MLPExtractor(obs_dim, hidden_dims, activation)

        feat_dim = self.features.output_dim

        # Policy head
        self.policy_mean = nn.Linear(feat_dim, action_dim)
        _init_weights(self.policy_mean, gain=0.01)

        # Learnable log-std
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init, dtype=torch.float32)
        )

        # Value head
        self.value_head = nn.Linear(feat_dim, 1)
        _init_weights(self.value_head, gain=1.0)

    def _forward_features(self, obs: Observation) -> torch.Tensor:
        if self.multimodal:
            if not isinstance(obs, dict):
                raise TypeError("Multimodal policy expects dict observations with image/state")
            if "image" not in obs or "state" not in obs:
                raise KeyError("Observation dict must contain 'image' and 'state'")

            image = obs["image"]
            state = obs["state"]

            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            else:
                image = image.float()
                if torch.max(image) > 1.0:
                    image = image / 255.0

            if image.dim() != 4:
                raise ValueError(f"Expected image batch rank 4, got shape {tuple(image.shape)}")
            if image.shape[1] != 3 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            elif image.shape[1] != 3:
                raise ValueError(
                    f"Expected image channels-first or channels-last RGB, got shape {tuple(image.shape)}"
                )

            state = state.float()
            state_feat = self.state_encoder(state)
            image_feat = self.image_encoder(image)
            fused = torch.cat([image_feat, state_feat], dim=-1)
            return self.features(fused)

        if isinstance(obs, dict):
            obs = obs["state"]
        return self.features(obs.float())

    def forward(self, obs: Observation) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._forward_features(obs)
        mean = self.policy_mean(features)
        value = self.value_head(features).squeeze(-1)
        return mean, value

    def get_action_and_value(
        self, obs: Observation, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action or evaluate given action under the current policy.

        Returns
        -------
        action : (B, action_dim)
        log_prob : (B,)
        entropy : (B,)
        value : (B,)
        """
        mean, value = self.forward(obs)
        std = torch.exp(torch.clamp(self.log_std, -20, 2))
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        # Clamp action to [-1, 1] for the environment
        action_clamped = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action_clamped, log_prob, entropy, value

    def get_value(self, obs: Observation) -> torch.Tensor:
        """Value-only forward pass (no policy computation)."""
        features = self._forward_features(obs)
        return self.value_head(features).squeeze(-1)

    def get_deterministic_action(self, obs: Observation) -> torch.Tensor:
        """Return the mean action (for evaluation)."""
        features = self._forward_features(obs)
        mean = self.policy_mean(features)
        return torch.clamp(mean, -1.0, 1.0)
