"""Reconstructors: upscale the LR image back to HR for the reconstruction loss.

The paper measures fidelity by upscaling the produced LR via *bicubic*
interpolation and comparing to the original HR. We therefore default to a
differentiable, MATLAB-compatible bicubic reconstructor. An optional jointly
trained EDSR-lite SR head is provided for the "downscale-for-SR" setting
(common SOTA practice) and is selected via config.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from adn.models.blocks import conv3x3, make_res_blocks, default_init
from adn.utils.imresize import imresize


class BicubicReconstructor(nn.Module):
    """Parameter-free differentiable bicubic upscaler (MATLAB-compatible)."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, lr: torch.Tensor, out_hw=None) -> torch.Tensor:
        if out_hw is not None:
            return imresize(lr, sizes=tuple(out_hw)).clamp(0, 1)
        return imresize(lr, scale=self.scale).clamp(0, 1)


class _Upsampler(nn.Sequential):
    def __init__(self, scale: int, channels: int) -> None:
        layers = []
        if (scale & (scale - 1)) == 0:  # power of two
            for _ in range(int(scale).bit_length() - 1):
                layers += [conv3x3(channels, 4 * channels), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [conv3x3(channels, 9 * channels), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"Unsupported SR scale {scale}")
        super().__init__(*layers)


class EDSRReconstructor(nn.Module):
    """Lightweight EDSR-style SR head for joint downscale-for-SR training."""

    def __init__(self, scale: int, channels: int = 64, num_blocks: int = 16,
                 res_scale: float = 0.1) -> None:
        super().__init__()
        self.head = conv3x3(3, channels)
        self.body = make_res_blocks(channels, num_blocks, res_scale=res_scale)
        self.body_tail = conv3x3(channels, channels)
        self.upsampler = _Upsampler(scale, channels)
        self.tail = conv3x3(channels, 3)
        default_init(self, scale=0.1)

    def forward(self, lr: torch.Tensor, out_hw=None) -> torch.Tensor:
        x = self.head(lr)
        res = self.body_tail(self.body(x)) + x
        x = self.upsampler(res)
        out = self.tail(x)
        if out_hw is not None and out.shape[-2:] != tuple(out_hw):
            out = nn.functional.interpolate(out, size=tuple(out_hw),
                                            mode="bilinear", align_corners=False)
        return out.clamp(0, 1)


def build_reconstructor(name: str, scale: int, **kwargs) -> nn.Module:
    name = (name or "bicubic").lower()
    if name == "bicubic":
        return BicubicReconstructor(scale)
    if name in ("edsr", "sr"):
        return EDSRReconstructor(scale, **kwargs)
    raise ValueError(f"Unknown reconstructor: {name}")
