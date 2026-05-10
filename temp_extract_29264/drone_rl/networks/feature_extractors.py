"""
Feature extractors: MLP, CNN, and combined architectures.
"""
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


def _init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class MLPExtractor(nn.Module):
    """
    MLP feature extractor with orthogonal initialization.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : tuple of int
        Hidden layer sizes.
    activation : str
        Activation function: "tanh", "relu", or "elu".
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 256),
                 activation: str = "tanh"):
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev
        self.net.apply(lambda m: _init_weights(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNExtractor(nn.Module):
    """
    CNN feature extractor for 64x64 RGB images.

    Architecture: 3 conv layers -> flatten -> linear.
    """

    def __init__(self, img_channels: int = 3, output_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute output dim
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, 64, 64)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(cnn_out, output_dim), nn.ReLU())
        self.output_dim = output_dim
        self.apply(lambda m: _init_weights(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) float32 normalised to [0, 1]."""
        return self.fc(self.cnn(x))


class VisionTransformerExtractor(nn.Module):
    """
    Lightweight ViT feature extractor for 64x64 RGB inputs.
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        output_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            img_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(lambda m: _init_weights(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) float32 normalised to [0, 1].
        """
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)     # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        encoded = self.encoder(tokens)
        cls_feat = self.norm(encoded[:, 0])
        return self.fc(cls_feat)


class CombinedExtractor(nn.Module):
    """
    Combined CNN + MLP extractor for multi-modal observations.
    """

    def __init__(self, state_dim: int, img_channels: int = 3,
                 cnn_output_dim: int = 256, mlp_hidden: int = 128):
        super().__init__()
        self.cnn = CNNExtractor(img_channels, cnn_output_dim)
        self.state_fc = nn.Sequential(nn.Linear(state_dim, mlp_hidden), nn.ReLU())
        self.output_dim = cnn_output_dim + mlp_hidden
        self.apply(lambda m: _init_weights(m))

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_feat = self.cnn(image)
        state_feat = self.state_fc(state)
        return torch.cat([img_feat, state_feat], dim=-1)
