"""Color-space conversions matching common SR benchmark conventions.

PSNR/SSIM in the SR literature are computed on the luminance (Y) channel of
YCbCr using the ITU-R BT.601 conversion used by MATLAB ``rgb2ycbcr``.
"""
from __future__ import annotations

import torch


# BT.601 (MATLAB rgb2ycbcr) matrix for inputs in [0, 1].
_RGB2Y = torch.tensor([65.481, 128.553, 24.966]) / 255.0
_Y_BIAS = 16.0 / 255.0


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image (..., 3, H, W) in [0,1] to YCbCr in [0,1]."""
    r, g, b = img[..., 0, :, :], img[..., 1, :, :], img[..., 2, :, :]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b)
    cb = 128.0 + (-37.797 * r - 74.203 * g + 112.0 * b)
    cr = 128.0 + (112.0 * r - 93.786 * g - 18.214 * b)
    out = torch.stack([y, cb, cr], dim=-3) / 255.0
    return out


def to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """Return the Y (luminance) channel in [0,1] from an RGB image (...,3,H,W).

    Grayscale (1-channel) inputs are returned unchanged.
    """
    if img.shape[-3] == 1:
        return img
    weight = _RGB2Y.to(img.device, img.dtype).view(3, 1, 1)
    y = (img[..., :3, :, :] * weight).sum(dim=-3, keepdim=True) + _Y_BIAS
    return y
