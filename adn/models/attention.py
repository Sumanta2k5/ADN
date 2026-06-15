"""Attention Refinement Module (ARM) = CBAM (Woo et al., ECCV 2018).

Implements the paper's equations exactly:

  Channel:  A_c(F) = sigma( W1(W0(GAP(F))) + W1(W0(GMP(F))) )
            F_r    = A_c(F) ⊙ F
  Spatial:  A_s(F) = sigma( f^{7x7}([AvgPool(F); MaxPool(F)]) )
            F_r    = A_s(F_r) ⊙ F_r
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention with a shared MLP over GAP and GMP descriptors."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # Shared MLP (W0 then W1), implemented with 1x1 convs.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.gap(x))
        max_out = self.mlp(self.gmp(x))
        attn = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)
        return attn


class SpatialAttention(nn.Module):
    """Spatial attention from concatenated channel-wise avg/max maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))  # (B, 1, H, W)
        return attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F_r = A_c(F) ⊙ F
        x = self.channel_attention(x) * x
        # F_r <- A_s(F_r) ⊙ F_r
        x = self.spatial_attention(x) * x
        return x


class ARM(nn.Module):
    """Attention Refinement Module.

    A residual CBAM wrapper: refined = F + cbam(F) when ``residual`` is True,
    which stabilizes training while preserving the paper's attention design.
    When ``enabled`` is False, ARM is an identity (used for the w/o-CBAM ablation).
    """

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7, residual: bool = True,
                 enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.residual = residual
        self.cbam = CBAM(channels, reduction, spatial_kernel) if enabled else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        refined = self.cbam(x)
        return x + refined if self.residual else refined
