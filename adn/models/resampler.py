"""Adaptive Resampling Layer (the core of ADN).

Implements the paper's deformable content-adaptive resampling:

  (1) HR projection of LR pixel (x, y):
          u = (x + 0.5) * s - 0.5 ;  v = (y + 0.5) * s - 0.5
  (2) Deformable tap positions for tap (i, j), with learned offsets:
          u' = u + (j - c_w) + dX(i,j)
          v' = v + (i - c_h) + dY(i,j)
      where the centering c = (k-1)/2 (standard) or k/2 (paper-literal).
  (3) Bilinear-sample HR at (u', v') and mix with the softmax kernel:
          I_LR(x, y) = sum_{i,j} K(i,j) * I_HR^bilinear(u', v')

The k*k taps are accumulated with a vectorized ``grid_sample`` loop (k is small,
typically 3), which is fully differentiable and memory-efficient.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_base_grid(out_h: int, out_w: int, scale: float, device, dtype):
    """Return base HR coordinates (u along W, v along H) for each LR pixel."""
    ys = torch.arange(out_h, device=device, dtype=dtype)
    xs = torch.arange(out_w, device=device, dtype=dtype)
    base_v = (ys + 0.5) * scale - 0.5     # (out_h,)
    base_u = (xs + 0.5) * scale - 0.5     # (out_w,)
    base_v = base_v.view(1, 1, out_h, 1)
    base_u = base_u.view(1, 1, 1, out_w)
    return base_u, base_v


def adaptive_resample(
    hr: torch.Tensor,
    kernels: torch.Tensor,
    offset_x: torch.Tensor,
    offset_y: torch.Tensor,
    scale: float,
    kernel_size: int,
    paper_centering: bool = False,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Differentiable deformable resampling.

    Args:
        hr: (B, C, H, W) high-resolution input.
        kernels: (B, k*k, Hl, Wl) softmax kernels.
        offset_x/offset_y: (B, k*k, Hl, Wl) per-tap offsets (HR pixel units).
        scale: downscaling factor s (HR/LR).
        kernel_size: k (m == n).
        paper_centering: if True use (i - k/2) as in the paper text; else the
            standard symmetric (i - (k-1)/2).
        padding_mode: grid_sample padding for out-of-range taps.
    Returns:
        (B, C, Hl, Wl) low-resolution output.
    """
    b, c, h, w = hr.shape
    k = kernel_size
    _, k2, hl, wl = kernels.shape
    assert k2 == k * k, "kernel channel mismatch"
    device, dtype = hr.device, hr.dtype

    base_u, base_v = _make_base_grid(hl, wl, scale, device, dtype)
    center = (k / 2.0) if paper_centering else ((k - 1) / 2.0)

    out = hr.new_zeros(b, c, hl, wl)
    idx = 0
    for i in range(k):        # row index -> vertical (v / H)
        for j in range(k):    # col index -> horizontal (u / W)
            du = (j - center) + offset_x[:, idx:idx + 1]   # (B,1,Hl,Wl)
            dv = (i - center) + offset_y[:, idx:idx + 1]
            su = base_u + du
            sv = base_v + dv
            # Normalize to [-1, 1] for align_corners=True grid_sample.
            gx = 2.0 * su / max(w - 1, 1) - 1.0
            gy = 2.0 * sv / max(h - 1, 1) - 1.0
            grid = torch.cat([gx, gy], dim=1).permute(0, 2, 3, 1)  # (B,Hl,Wl,2)
            sampled = F.grid_sample(
                hr, grid, mode="bilinear",
                padding_mode=padding_mode, align_corners=True,
            )
            out = out + kernels[:, idx:idx + 1] * sampled
            idx += 1
    return out


class AdaptiveResamplingLayer(nn.Module):
    def __init__(self, kernel_size: int = 3, paper_centering: bool = False,
                 padding_mode: str = "border") -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.paper_centering = paper_centering
        self.padding_mode = padding_mode

    def forward(self, hr, kernels, offset_x, offset_y, scale) -> torch.Tensor:
        return adaptive_resample(
            hr, kernels, offset_x, offset_y, scale,
            self.kernel_size, self.paper_centering, self.padding_mode,
        )
