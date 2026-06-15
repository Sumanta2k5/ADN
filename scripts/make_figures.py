"""Generate the paper's qualitative figures.

For selected benchmark images, produces:
  * Comparison grids: HR | Bicubic-LR | ADN-LR  and  HR | Bicubic-SR | ADN-SR
    (with PSNR/SSIM annotations) -- reproduces Figs. 3-6.
  * Kernel entropy heatmaps -- reproduces the kernel-heatmap discussion (Sec. V).
  * Offset quiver fields -- visualizes the deformable sampling (OEM).

Usage:
    python -m scripts.make_figures --config configs/adn_x4.yaml \
        --checkpoint experiments/adn_x4/checkpoints/best.pth \
        --dataset Urban100 --num 4 --out figures/adn_x4
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

from adn.data.transforms import ensure_divisible, to_tensor
from adn.metrics import calculate_psnr, calculate_ssim
from adn.models import build_adn
from adn.utils import load_config
from adn.utils.checkpoint import load_checkpoint
from adn.utils.imresize import imresize
from adn.utils.visualize import (
    save_comparison_grid,
    save_kernel_heatmap,
    save_offset_field,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _list(root):
    out = []
    for ext in IMG_EXTS:
        import glob
        out += glob.glob(os.path.join(root, f"*{ext}"))
    return sorted(out)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="Folder of HR images.")
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--out", default="figures")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    scale = int(cfg.get("scale", 2))
    align = cfg.get("model", {}).get("feature_downscale", 8) * scale

    model = build_adn(cfg).to(args.device).eval()
    load_checkpoint(args.checkpoint, model, map_location=args.device, strict=False)

    files = _list(args.dataset)[: args.num]
    os.makedirs(args.out, exist_ok=True)

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        img = ensure_divisible(np.asarray(Image.open(f).convert("RGB")), align)
        hr = to_tensor(img).unsqueeze(0).to(args.device)

        out = model(hr, return_params=True)
        lr_adn, recon_adn = out["lr"], out["recon"]

        # Baselines: bicubic downscale + bicubic upscale.
        lr_bic = imresize(hr, sizes=lr_adn.shape[-2:]).clamp(0, 1)
        recon_bic = imresize(lr_bic, sizes=hr.shape[-2:]).clamp(0, 1)

        p_bic = calculate_psnr(recon_bic, hr, crop_border=scale)
        s_bic = calculate_ssim(recon_bic, hr, crop_border=scale)
        p_adn = calculate_psnr(recon_adn, hr, crop_border=scale)
        s_adn = calculate_ssim(recon_adn, hr, crop_border=scale)

        # LR comparison.
        save_comparison_grid(
            {"HR": hr[0], "Bicubic-LR": lr_bic[0], "ADN-LR (Ours)": lr_adn[0]},
            os.path.join(args.out, f"{name}_LR_compare.png"),
        )
        # SR/reconstruction comparison with metrics in titles.
        save_comparison_grid(
            {
                "HR": hr[0],
                f"Bicubic {p_bic:.2f}/{s_bic:.3f}": recon_bic[0],
                f"ADN {p_adn:.2f}/{s_adn:.3f}": recon_adn[0],
            },
            os.path.join(args.out, f"{name}_SR_compare.png"),
        )
        # Kernel heatmap + offset field.
        save_kernel_heatmap(out["kernels"][0], os.path.join(args.out, f"{name}_kernel.png"))
        save_offset_field(out["offset_x"][0], out["offset_y"][0],
                          os.path.join(args.out, f"{name}_offset.png"))
        print(f"[fig] {name}: Bicubic {p_bic:.2f}/{s_bic:.3f} | ADN {p_adn:.2f}/{s_adn:.3f}")

    print(f"Figures saved to {args.out}/")


if __name__ == "__main__":
    main()
