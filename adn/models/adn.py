"""ADN: Adaptive Image Downscaling Network (full model).

Forward pipeline (paper Sec. III):

    HR --FeatureExtractor--> F (H/down)
       --ARM (CBAM)--------> F_r
       --resize to H/s-----> (KGM -> kernels, OEM -> offsets)
       --AdaptiveResampling-> LR
    LR --Reconstructor------> recon (for loss / PSNR-SSIM evaluation)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from adn.models.attention import ARM
from adn.models.feature_extractor import FeatureExtractor
from adn.models.kernel_offset import KernelGenerationModule, OffsetEstimationModule
from adn.models.resampler import AdaptiveResamplingLayer
from adn.models.reconstructor import build_reconstructor


class ADN(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        in_channels: int = 3,
        channels: int = 64,
        kernel_size: int = 3,
        feature_downscale: int = 8,
        num_feature_blocks: int = 8,
        kernel_depth: int = 5,
        offset_depth: int = 3,
        cbam_reduction: int = 16,
        spatial_kernel: int = 7,
        offset_scale: float = 1.0,
        kernel_temperature: float = 1.0,
        res_scale: float = 1.0,
        use_cbam: bool = True,
        use_resnet: bool = True,
        arm_residual: bool = True,
        paper_centering: bool = False,
        padding_mode: str = "border",
        reconstructor: str = "bicubic",
        recon_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size

        self.feature_extractor = FeatureExtractor(
            in_channels=in_channels,
            channels=channels,
            num_res_blocks=num_feature_blocks,
            downscale=feature_downscale,
            res_scale=res_scale,
            use_resnet=use_resnet,
        )
        self.arm = ARM(
            channels=channels,
            reduction=cbam_reduction,
            spatial_kernel=spatial_kernel,
            residual=arm_residual,
            enabled=use_cbam,
        )
        self.kgm = KernelGenerationModule(
            channels=channels, kernel_size=kernel_size, depth=kernel_depth,
            res_scale=res_scale, temperature=kernel_temperature,
        )
        self.oem = OffsetEstimationModule(
            channels=channels, kernel_size=kernel_size, depth=offset_depth,
            res_scale=res_scale, offset_scale=offset_scale,
        )
        self.resampler = AdaptiveResamplingLayer(
            kernel_size=kernel_size, paper_centering=paper_centering,
            padding_mode=padding_mode,
        )
        self.reconstructor = build_reconstructor(
            reconstructor, scale, **(recon_kwargs or {})
        )

    # ------------------------------------------------------------------ #
    def downscale(self, hr: torch.Tensor, return_params: bool = False):
        """Produce the LR image from HR (the downscaling operator)."""
        b, c, h, w = hr.shape
        hl = round(h / self.scale)
        wl = round(w / self.scale)

        feat = self.feature_extractor(hr)
        feat = self.arm(feat)

        kernels = self.kgm(feat, (hl, wl))
        dx, dy = self.oem(feat, (hl, wl))
        lr = self.resampler(hr, kernels, dx, dy, self.scale)
        lr = lr.clamp(0, 1)

        if return_params:
            return lr, {"kernels": kernels, "offset_x": dx, "offset_y": dy}
        return lr

    def reconstruct(self, lr: torch.Tensor, out_hw=None) -> torch.Tensor:
        return self.reconstructor(lr, out_hw=out_hw)

    def forward(self, hr: torch.Tensor, return_params: bool = False
                ) -> Dict[str, torch.Tensor]:
        out = self.downscale(hr, return_params=return_params)
        if return_params:
            lr, params = out
        else:
            lr, params = out, {}
        recon = self.reconstruct(lr, out_hw=hr.shape[-2:])
        result = {"lr": lr, "recon": recon}
        result.update(params)
        return result


def build_adn(cfg) -> ADN:
    """Build an ADN model from a (possibly nested) config object/dict."""
    m = cfg["model"] if "model" in cfg else cfg
    return ADN(
        scale=int(cfg.get("scale", m.get("scale", 2))),
        in_channels=m.get("in_channels", 3),
        channels=m.get("channels", 64),
        kernel_size=m.get("kernel_size", 3),
        feature_downscale=m.get("feature_downscale", 8),
        num_feature_blocks=m.get("num_feature_blocks", 8),
        kernel_depth=m.get("kernel_depth", 5),
        offset_depth=m.get("offset_depth", 3),
        cbam_reduction=m.get("cbam_reduction", 16),
        spatial_kernel=m.get("spatial_kernel", 7),
        offset_scale=m.get("offset_scale", 1.0),
        kernel_temperature=m.get("kernel_temperature", 1.0),
        res_scale=m.get("res_scale", 1.0),
        use_cbam=m.get("use_cbam", True),
        use_resnet=m.get("use_resnet", True),
        arm_residual=m.get("arm_residual", True),
        paper_centering=m.get("paper_centering", False),
        padding_mode=m.get("padding_mode", "border"),
        reconstructor=m.get("reconstructor", "bicubic"),
        recon_kwargs=m.get("recon_kwargs", None),
    )
