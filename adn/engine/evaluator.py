"""Evaluation engine: runs ADN over benchmark loaders and computes metrics.

For each HR image:
  1. LR  = ADN.downscale(HR)
  2. SR  = ADN.reconstruct(LR)         (bicubic by default, or jointly trained SR)
  3. PSNR/SSIM(SR, HR) on Y channel with ``crop_border = scale``
  4. NIQE(LR)  (no-reference perceptual quality of the downscaled image)

Optionally saves LR / SR / comparison images for qualitative figures.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from adn.metrics import calculate_niqe, calculate_psnr, calculate_ssim, niqe_available


def _save_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (tensor.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)
    arr = arr.round().astype(np.uint8)
    Image.fromarray(arr).save(path)


class Evaluator:
    def __init__(self, scale: int, test_y_channel: bool = True,
                 compute_niqe: bool = True, device: str = "cuda") -> None:
        self.scale = scale
        self.test_y_channel = test_y_channel
        self.compute_niqe = compute_niqe and niqe_available()
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, loader, name: str = "val",
                 save_dir: Optional[str] = None, max_images: Optional[int] = None
                 ) -> Dict[str, float]:
        model.eval()
        psnrs, ssims, niqes = [], [], []
        per_image = []
        for i, batch in enumerate(tqdm(loader, desc=f"eval[{name}]", leave=False)):
            if max_images is not None and i >= max_images:
                break
            hr = batch["hr"].to(self.device, non_blocking=True)
            out = model(hr)
            lr, recon = out["lr"], out["recon"]

            psnr = calculate_psnr(recon, hr, crop_border=self.scale,
                                  test_y_channel=self.test_y_channel)
            ssim_v = calculate_ssim(recon, hr, crop_border=self.scale,
                                    test_y_channel=self.test_y_channel)
            psnrs.append(psnr)
            ssims.append(ssim_v)
            niqe_v = float("nan")
            if self.compute_niqe:
                niqe_v = calculate_niqe(lr[0], device=self.device)
                if not np.isnan(niqe_v):
                    niqes.append(niqe_v)

            img_name = batch.get("name", [f"{i:04d}"])[0]
            per_image.append({"name": img_name, "psnr": psnr,
                              "ssim": ssim_v, "niqe": niqe_v})

            if save_dir is not None:
                _save_image(lr[0], os.path.join(save_dir, name, f"{img_name}_LR.png"))
                _save_image(recon[0], os.path.join(save_dir, name, f"{img_name}_SR.png"))

        results = {
            "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
            "ssim": float(np.mean(ssims)) if ssims else float("nan"),
            "niqe": float(np.mean(niqes)) if niqes else float("nan"),
            "num_images": len(psnrs),
        }
        results["_per_image"] = per_image
        return results


@torch.no_grad()
def evaluate_model(model, val_loaders: Dict, scale: int, device: str = "cuda",
                   save_dir: Optional[str] = None, test_y_channel: bool = True,
                   compute_niqe: bool = True) -> Dict[str, Dict[str, float]]:
    evaluator = Evaluator(scale, test_y_channel, compute_niqe, device)
    all_results = {}
    for name, loader in val_loaders.items():
        all_results[name] = evaluator.evaluate(model, loader, name=name,
                                                save_dir=save_dir)
    return all_results
