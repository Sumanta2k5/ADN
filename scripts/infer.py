"""Run ADN downscaling on arbitrary images (single image or a folder).

Usage:
    python -m scripts.infer --config configs/adn_x4.yaml \
        --checkpoint experiments/adn_x4/checkpoints/best.pth \
        --input path/to/img_or_dir --output out_dir
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from adn.data.transforms import ensure_divisible, to_tensor
from adn.models import build_adn
from adn.utils import load_config
from adn.utils.checkpoint import load_checkpoint

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _gather(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(path, f"*{ext}"))
        files += glob.glob(os.path.join(path, f"*{ext.upper()}"))
    return sorted(files)


def _save(t, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="ADN inference")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="infer_out")
    p.add_argument("--save-recon", action="store_true", help="Also save bicubic/SR reconstruction.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    scale = int(cfg.get("scale", 2))
    feat_down = cfg.get("model", {}).get("feature_downscale", 8)
    align = feat_down * scale

    model = build_adn(cfg).to(args.device).eval()
    load_checkpoint(args.checkpoint, model, map_location=args.device, strict=False)

    files = _gather(args.input)
    if not files:
        raise FileNotFoundError(f"No images found at {args.input}")
    print(f"Processing {len(files)} image(s) at x{scale} ...")

    for f in files:
        img = ensure_divisible(np.asarray(Image.open(f).convert("RGB")), align)
        hr = to_tensor(img).unsqueeze(0).to(args.device)
        out = model(hr)
        name = os.path.splitext(os.path.basename(f))[0]
        _save(out["lr"][0], os.path.join(args.output, f"{name}_LRx{scale}.png"))
        if args.save_recon:
            _save(out["recon"][0], os.path.join(args.output, f"{name}_reconx{scale}.png"))
    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
