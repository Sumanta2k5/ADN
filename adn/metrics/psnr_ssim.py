"""PSNR / SSIM following SR benchmark conventions.

Conventions (matching CAR / EDSR / RCAN evaluation protocols):
  * compute on the Y (luminance) channel of YCbCr by default,
  * crop ``scale`` pixels from each border before computing,
  * images are compared in [0, 1].

These produce numbers directly comparable to the tables in the paper.
"""
from __future__ import annotations

import torch

from adn.losses.ssim import ssim as _ssim_engine
from adn.utils.color import to_y_channel


def _prepare(img1: torch.Tensor, img2: torch.Tensor, crop_border: int,
             test_y_channel: bool):
    assert img1.shape == img2.shape, f"shape mismatch {img1.shape} vs {img2.shape}"
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    img1 = img1.clamp(0, 1).to(torch.float64)
    img2 = img2.clamp(0, 1).to(torch.float64)
    if test_y_channel and img1.shape[1] == 3:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    if crop_border > 0:
        img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    return img1, img2


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, crop_border: int = 0,
                   test_y_channel: bool = True, max_val: float = 1.0) -> float:
    a, b = _prepare(img1, img2, crop_border, test_y_channel)
    mse = torch.mean((a - b) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-12)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return float(psnr.mean().item())


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, crop_border: int = 0,
                   test_y_channel: bool = True) -> float:
    a, b = _prepare(img1, img2, crop_border, test_y_channel)
    val = _ssim_engine(a, b, window_size=11, sigma=1.5, data_range=1.0)
    return float(val.item())


class PSNR:
    def __init__(self, crop_border: int = 0, test_y_channel: bool = True) -> None:
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel

    def __call__(self, img1, img2) -> float:
        return calculate_psnr(img1, img2, self.crop_border, self.test_y_channel)


class SSIMMetric:
    def __init__(self, crop_border: int = 0, test_y_channel: bool = True) -> None:
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel

    def __call__(self, img1, img2) -> float:
        return calculate_ssim(img1, img2, self.crop_border, self.test_y_channel)
