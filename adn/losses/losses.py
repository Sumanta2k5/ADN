"""ADN hybrid training objective.

Paper Eq. (4):   L = lambda1 * L_MSE + lambda2 * L_SSIM
with lambda1 = 0.55, lambda2 = 0.45 by default, where:

  * L_MSE  : MSE between the reconstruction (LR upscaled) and HR.
  * L_SSIM : (1 - SSIM) between the reconstruction and HR.

Optional regularizers (SOTA practice, disabled by default to stay faithful):
  * guidance:  keep the LR close to the bicubic-downscaled HR
               (stabilizes/keeps LR natural; weight ``lambda_guidance``).
  * tv:        total-variation smoothness on the LR.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from adn.losses.ssim import ssim
from adn.utils.imresize import imresize


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


class HybridDownscaleLoss(nn.Module):
    def __init__(
        self,
        lambda_mse: float = 0.55,
        lambda_ssim: float = 0.45,
        lambda_guidance: float = 0.0,
        lambda_tv: float = 0.0,
        scale: int = 2,
        ssim_window: int = 11,
    ) -> None:
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_guidance = lambda_guidance
        self.lambda_tv = lambda_tv
        self.scale = scale
        self.ssim_window = ssim_window

    def forward(self, outputs: Dict[str, torch.Tensor], hr: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        recon = outputs["recon"]
        lr = outputs["lr"]

        l_mse = F.mse_loss(recon, hr)
        l_ssim = 1.0 - ssim(recon, hr, window_size=self.ssim_window)

        total = self.lambda_mse * l_mse + self.lambda_ssim * l_ssim
        logs = {"loss/mse": l_mse.detach(), "loss/ssim": l_ssim.detach()}

        if self.lambda_guidance > 0:
            with torch.no_grad():
                lr_ref = imresize(hr, sizes=lr.shape[-2:]).clamp(0, 1)
            l_guide = F.l1_loss(lr, lr_ref)
            total = total + self.lambda_guidance * l_guide
            logs["loss/guidance"] = l_guide.detach()

        if self.lambda_tv > 0:
            l_tv = total_variation(lr)
            total = total + self.lambda_tv * l_tv
            logs["loss/tv"] = l_tv.detach()

        logs["loss/total"] = total.detach()
        return {"total": total, "logs": logs}


def build_loss(cfg) -> HybridDownscaleLoss:
    l = cfg.get("loss", {})
    return HybridDownscaleLoss(
        lambda_mse=l.get("lambda_mse", 0.55),
        lambda_ssim=l.get("lambda_ssim", 0.45),
        lambda_guidance=l.get("lambda_guidance", 0.0),
        lambda_tv=l.get("lambda_tv", 0.0),
        scale=int(cfg.get("scale", 2)),
        ssim_window=l.get("ssim_window", 11),
    )
