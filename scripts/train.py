"""Train an ADN model.

Usage:
    python -m scripts.train --config configs/adn_x2.yaml
    python -m scripts.train --config configs/adn_x4.yaml --opts train.total_iters=100000

Multi-GPU (single node) via torchrun is supported transparently for the data
(single-process training is the default; DDP can be added by wrapping the model).
"""
from __future__ import annotations

import argparse
import os

import torch

from adn.data import build_dataloaders
from adn.engine import Trainer
from adn.losses import build_loss
from adn.models import build_adn
from adn.utils import get_logger, load_config, set_seed
from adn.utils.config import override_from_dotlist


def parse_args():
    p = argparse.ArgumentParser(description="Train ADN")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--work-dir", default=None, help="Override output directory.")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opts", nargs="*", default=[], help="key=value overrides.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = override_from_dotlist(cfg, args.opts)

    work_dir = args.work_dir or os.path.join(
        cfg.get("output_dir", "experiments"),
        cfg.get("name", os.path.splitext(os.path.basename(args.config))[0]),
    )
    os.makedirs(work_dir, exist_ok=True)

    set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False))
    logger = get_logger("adn", os.path.join(work_dir, "train.log"))
    logger.info(f"Config: {args.config}")
    logger.info(f"Work dir: {work_dir}")
    logger.info(f"Device: {args.device}")

    train_loader, val_loaders = build_dataloaders(cfg)
    if train_loader is None:
        raise RuntimeError("No training data configured (data.train_root).")
    logger.info(f"Train images: {len(train_loader.dataset)} | "
                f"val sets: {list(val_loaders.keys())}")

    model = build_adn(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params/1e6:.3f} M")

    criterion = build_loss(cfg)

    trainer = Trainer(model, criterion, train_loader, val_loaders, cfg,
                      device=args.device, work_dir=work_dir)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
