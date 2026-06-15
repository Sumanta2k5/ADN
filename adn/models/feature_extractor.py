"""Deep convolutional feature extractor.

Encodes an HR image into a latent feature map at H/down x W/down (paper: /8)
using a conv stem, residual blocks, and strided downsampling stages.
The ``use_resnet`` flag enables the w/o-ResNet ablation (Table V).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from adn.models.blocks import conv3x3, make_res_blocks, default_init


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        num_res_blocks: int = 8,
        downscale: int = 8,
        res_scale: float = 1.0,
        use_resnet: bool = True,
        act: str = "relu",
    ) -> None:
        super().__init__()
        assert downscale >= 1 and (downscale & (downscale - 1) == 0), (
            "downscale must be a power of two."
        )
        self.downscale = downscale
        self.use_resnet = use_resnet

        self.stem = conv3x3(in_channels, channels)

        num_stages = int(math.log2(downscale))
        stages = []
        for _ in range(num_stages):
            if use_resnet:
                stages.append(make_res_blocks(channels, max(1, num_res_blocks // max(1, num_stages)),
                                              res_scale=res_scale, act=act))
            else:
                stages.append(nn.Sequential(conv3x3(channels, channels), _act(act)))
            # Strided downsample by 2.
            stages.append(nn.Sequential(conv3x3(channels, channels, stride=2), _act(act)))
        self.encoder = nn.Sequential(*stages)

        if use_resnet:
            self.body = make_res_blocks(channels, num_res_blocks, res_scale=res_scale, act=act)
        else:
            self.body = nn.Sequential(conv3x3(channels, channels), _act(act))

        self.fuse = conv3x3(channels, channels)
        default_init(self, scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        x = self.fuse(self.body(x))
        return x


def _act(name: str) -> nn.Module:
    from adn.models.blocks import _make_act
    return _make_act(name)
