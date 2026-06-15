"""Evaluate a trained ADN model on benchmark datasets and dump metrics to JSON.

Usage:
    python -m scripts.test --config configs/adn_x4.yaml \
        --checkpoint experiments/adn_x4/checkpoints/best.pth \
        --save-dir results/adn_x4
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from adn.data import build_dataloaders
from adn.engine import Evaluator
from adn.models import build_adn
from adn.utils import get_logger, load_config, set_seed
from adn.utils.checkpoint import load_checkpoint
from adn.utils.config import override_from_dotlist


def parse_args():
    p = argparse.ArgumentParser(description="Test/evaluate ADN")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save-dir", default=None, help="Where to save LR/SR images + metrics.")
    p.add_argument("--use-ema", action="store_true", help="Load EMA weights if present.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opts", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = override_from_dotlist(cfg, args.opts)
    set_seed(cfg.get("seed", 42))

    save_dir = args.save_dir or os.path.join("results",
                                             cfg.get("name", "adn"))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger("adn-test", os.path.join(save_dir, "test.log"))

    _, val_loaders = build_dataloaders(cfg)
    if not val_loaders:
        raise RuntimeError("No validation/test sets configured (data.val_roots).")

    model = build_adn(cfg).to(args.device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=args.device, strict=False)
    if args.use_ema and "extra" in ckpt and ckpt["extra"] and "ema" in ckpt["extra"]:
        model.load_state_dict(ckpt["extra"]["ema"])
        logger.info("Loaded EMA weights.")
    logger.info(f"Loaded checkpoint {args.checkpoint} (step {ckpt.get('step')}).")

    evaluator = Evaluator(
        scale=int(cfg.get("scale", 2)),
        test_y_channel=cfg.get("eval", {}).get("test_y_channel", True),
        compute_niqe=cfg.get("eval", {}).get("compute_niqe", True),
        device=args.device,
    )

    summary = {}
    for name, loader in val_loaders.items():
        res = evaluator.evaluate(model, loader, name=name, save_dir=save_dir)
        per_image = res.pop("_per_image")
        summary[name] = res
        logger.info(f"[{name}] PSNR {res['psnr']:.4f} | SSIM {res['ssim']:.4f} "
                    f"| NIQE {res['niqe']:.4f} | n={res['num_images']}")
        with open(os.path.join(save_dir, f"{name}_per_image.json"), "w") as f:
            json.dump(per_image, f, indent=2)

    out = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "scale": int(cfg.get("scale", 2)),
        "results": summary,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved metrics to {os.path.join(save_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
