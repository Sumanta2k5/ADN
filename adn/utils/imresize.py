"""MATLAB-compatible ``imresize`` (bicubic/bilinear) in PyTorch.

SR benchmarks (Set5/Set14/BSD100/Urban100/DIV2K) are conventionally produced
with MATLAB's ``imresize`` using antialiasing. PyTorch's ``F.interpolate`` does
NOT match MATLAB bicubic, which biases PSNR/SSIM. This module reproduces
MATLAB's algorithm (cubic kernel a=-0.5, antialiasing) so that our LR/HR pairs
and the bicubic up/down references are directly comparable to published numbers.

Differentiable: operations are pure tensor ops, so this can be used inside the
training loop (e.g. for the bicubic-guidance regularizer).

Reference: Sun & Chen, "Learned Image Downscaling..." released a widely-used
PyTorch port of MATLAB imresize; the math below follows MATLAB's source.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def _cubic(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax ** 2
    ax3 = ax ** 3
    cond1 = (ax <= 1).to(x.dtype)
    cond2 = ((ax > 1) & (ax <= 2)).to(x.dtype)
    f1 = 1.5 * ax3 - 2.5 * ax2 + 1.0
    f2 = -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0
    return f1 * cond1 + f2 * cond2


def _contributions(in_len: int, out_len: int, scale: float, kernel_width: float,
                   antialiasing: bool, device, dtype):
    if scale < 1 and antialiasing:
        kernel_width = kernel_width / scale

    x = torch.arange(1, out_len + 1, device=device, dtype=dtype)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = torch.floor(u - kernel_width / 2.0)
    p = int(math.ceil(kernel_width)) + 2

    indices = left.view(out_len, 1) + torch.arange(0, p, device=device, dtype=dtype).view(1, p)
    distance = u.view(out_len, 1) - indices

    if scale < 1 and antialiasing:
        weights = scale * _cubic(distance * scale)
    else:
        weights = _cubic(distance)

    weights = weights / weights.sum(dim=1, keepdim=True)

    indices = indices.long()
    # Mirror-reflect indices into valid range (MATLAB border handling).
    indices = indices - 1  # to 0-based
    aux = torch.cat([
        torch.arange(in_len, device=device),
        torch.arange(in_len - 1, -1, -1, device=device),
    ])
    period = 2 * in_len
    indices = aux[torch.remainder(indices, period)]

    # Drop columns whose weights are all zero.
    nonzero = weights.abs().sum(dim=0) > 1e-12
    weights = weights[:, nonzero]
    indices = indices[:, nonzero]
    return weights, indices


def _resize_along_dim(img: torch.Tensor, dim: int, weights: torch.Tensor,
                      indices: torch.Tensor) -> torch.Tensor:
    # img: (C, H, W). Resize along spatial dim (1=H, 2=W).
    img_t = img.transpose(dim, -1)          # bring target dim to last
    gathered = img_t[..., indices]          # (..., out_len, taps)
    w = weights.to(img.dtype)               # (out_len, taps)
    out = (gathered * w).sum(dim=-1)        # (..., out_len)
    return out.transpose(dim, -1)


def imresize(
    img: torch.Tensor,
    scale: Optional[float] = None,
    sizes: Optional[tuple] = None,
    kernel: str = "cubic",
    antialiasing: bool = True,
) -> torch.Tensor:
    """Resize an image tensor exactly like MATLAB ``imresize``.

    Args:
        img: (C, H, W) or (B, C, H, W) float tensor (any range).
        scale: scalar resize factor (e.g. 0.5 to downscale by 2, 2 to upscale).
        sizes: explicit (out_h, out_w); overrides ``scale`` if given.
        kernel: only ``"cubic"`` is supported (the SR-standard kernel).
        antialiasing: MATLAB-style antialiasing on downscale.
    """
    if kernel != "cubic":
        raise NotImplementedError("Only the cubic kernel is supported.")

    squeeze_batch = img.dim() == 3
    if squeeze_batch:
        img = img.unsqueeze(0)
    b, c, h, w = img.shape
    device, dtype = img.device, torch.float64

    if sizes is None:
        assert scale is not None, "Provide either scale or sizes."
        out_h = int(math.ceil(h * scale))
        out_w = int(math.ceil(w * scale))
        scale_h = scale_w = float(scale)
    else:
        out_h, out_w = sizes
        scale_h = out_h / h
        scale_w = out_w / w

    img64 = img.to(dtype)
    kw = 4.0
    wh, ih = _contributions(h, out_h, scale_h, kw, antialiasing, device, dtype)
    ww, iw = _contributions(w, out_w, scale_w, kw, antialiasing, device, dtype)

    out = torch.empty(b, c, out_h, out_w, device=device, dtype=dtype)
    for n in range(b):
        tmp = _resize_along_dim(img64[n], 1, wh, ih)  # resize H
        out[n] = _resize_along_dim(tmp, 2, ww, iw)    # resize W

    out = out.to(img.dtype)
    if squeeze_batch:
        out = out.squeeze(0)
    return out
