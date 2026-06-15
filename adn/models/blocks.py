"""Common building blocks: convolutions, residual blocks, initialization."""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3x3(in_ch: int, out_ch: int, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=bias)


class ResidualBlock(nn.Module):
    """Pre-activation-free residual block (EDSR style): Conv-Act-Conv + skip.

    Uses a residual scaling factor for training stability with deep stacks.
    """

    def __init__(self, channels: int, res_scale: float = 1.0,
                 act: str = "relu", norm: bool = False) -> None:
        super().__init__()
        self.conv1 = conv3x3(channels, channels)
        self.conv2 = conv3x3(channels, channels)
        self.norm1 = nn.BatchNorm2d(channels) if norm else nn.Identity()
        self.norm2 = nn.BatchNorm2d(channels) if norm else nn.Identity()
        self.act = _make_act(act)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return x + out * self.res_scale


def _make_act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("lrelu", "leaky_relu"):
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def make_res_blocks(channels: int, num_blocks: int, res_scale: float = 1.0,
                    act: str = "relu", norm: bool = False) -> nn.Sequential:
    return nn.Sequential(*[
        ResidualBlock(channels, res_scale=res_scale, act=act, norm=norm)
        for _ in range(num_blocks)
    ])


def default_init(module: nn.Module, scale: float = 0.1) -> None:
    """Kaiming init for convs with a small scaling (EDSR-style stability)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def zero_init(conv: nn.Conv2d) -> None:
    """Zero-initialize a conv (used so offsets start at the regular grid)."""
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
