"""Differentiable SSIM / MS-SSIM (Wang et al., 2004) for loss and metrics.

Operates on tensors in [0, 1]. Uses a Gaussian window and the standard
constants C1=(0.01)^2, C2=(0.03)^2. Shared by both the training loss
(``1 - SSIM``) and evaluation (the *metric* version on the Y channel lives in
``adn.metrics``; this module is the core engine).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)               # (W, 1)
    window_2d = (window_1d @ window_1d.t())  # (W, W)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _ssim_map(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor,
              window_size: int, data_range: float, channels: int):
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    cs_map = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    ssim_map = ((2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs_map
    return ssim_map, cs_map


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5, data_range: float = 1.0,
         size_average: bool = True) -> torch.Tensor:
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    ssim_map, _ = _ssim_map(x, y, window, window_size, data_range, channels)
    return ssim_map.mean() if size_average else ssim_map.mean([1, 2, 3])


def ms_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11,
            sigma: float = 1.5, data_range: float = 1.0,
            weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                               device=x.device, dtype=x.dtype)
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    levels = weights.numel()
    mcs = []
    for i in range(levels):
        ssim_map, cs_map = _ssim_map(x, y, window, window_size, data_range, channels)
        if i < levels - 1:
            mcs.append(cs_map.mean().clamp_min(1e-6))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    ssim_val = ssim_map.mean().clamp_min(1e-6)
    mcs_t = torch.stack(mcs + [ssim_val])
    return torch.prod(mcs_t ** weights)


class SSIM(nn.Module):
    """SSIM as a module; returns the mean SSIM in [0, 1]."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 1.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ssim(x, y, self.window_size, self.sigma, self.data_range)
