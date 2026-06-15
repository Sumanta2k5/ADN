"""Kernel Generation Module (KGM) and Offset Estimation Module (OEM).

Both take the ARM-refined feature map (at H/down) and produce, at the LR
resolution (H/scale), per-pixel parameters for the adaptive resampling layer:

  * KGM -> resampling kernels K of shape (B, k*k, H/s, W/s), softmax-normalized
           over the k*k taps so each kernel sums to 1 (brightness-preserving).
  * OEM -> offsets (dX, dY), each (B, k*k, H/s, W/s); final layer zero-init so
           sampling starts from the regular deformable grid.

Features are resized to the LR resolution before the residual trunk. The number
of residual blocks in the KGM trunk is the "kernel depth" ablated in the paper.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from adn.models.blocks import conv3x3, make_res_blocks, default_init, zero_init, _make_act


class _ParamHead(nn.Module):
    """Shared trunk used by KGM/OEM: resize-to-LR + residual blocks."""

    def __init__(self, channels: int, depth: int, res_scale: float = 1.0,
                 act: str = "relu") -> None:
        super().__init__()
        self.pre = conv3x3(channels, channels)
        self.act = _make_act(act)
        self.trunk = make_res_blocks(channels, depth, res_scale=res_scale, act=act)

    def forward(self, feat: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        if feat.shape[-2:] != out_hw:
            feat = F.interpolate(feat, size=out_hw, mode="bilinear", align_corners=False)
        feat = self.act(self.pre(feat))
        return self.trunk(feat)


class KernelGenerationModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, depth: int = 5,
                 res_scale: float = 1.0, act: str = "relu",
                 temperature: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.temperature = temperature
        self.head = _ParamHead(channels, depth, res_scale, act)
        self.out = conv3x3(channels, self.k2)
        default_init(self.head, scale=0.1)
        # Zero-init -> uniform softmax kernel ~ box average at start (stable).
        zero_init(self.out)

    def forward(self, feat: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.head(feat, out_hw)
        logits = self.out(x) / self.temperature           # (B, k*k, Hl, Wl)
        kernels = F.softmax(logits, dim=1)
        return kernels


class OffsetEstimationModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, depth: int = 3,
                 res_scale: float = 1.0, act: str = "relu",
                 offset_scale: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.offset_scale = offset_scale
        self.head = _ParamHead(channels, depth, res_scale, act)
        self.out = conv3x3(channels, 2 * self.k2)
        default_init(self.head, scale=0.1)
        # Zero-init -> start exactly on the regular sampling grid.
        zero_init(self.out)

    def forward(self, feat: torch.Tensor, out_hw: Tuple[int, int]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.head(feat, out_hw)
        offsets = self.out(x) * self.offset_scale          # (B, 2*k*k, Hl, Wl)
        dx, dy = torch.split(offsets, self.k2, dim=1)
        return dx, dy
