"""Training engine for ADN.

Features:
  * AMP mixed precision, gradient clipping, EMA (optional).
  * Cosine / step / multistep LR schedules.
  * Periodic validation with best-checkpoint tracking (by PSNR on a chosen set).
  * TensorBoard + file logging, resumable checkpoints.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Dict, Optional

import torch

from adn.engine.evaluator import Evaluator
from adn.utils.checkpoint import load_checkpoint, save_checkpoint
from adn.utils.logger import MetricTracker, get_logger, get_tensorboard_writer


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def build_optimizer(model, cfg) -> torch.optim.Optimizer:
    t = cfg["train"]
    name = t.get("optimizer", "adam").lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=t.get("lr", 2e-4),
                                betas=tuple(t.get("betas", (0.9, 0.999))),
                                weight_decay=t.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=t.get("lr", 2e-4),
                                 betas=tuple(t.get("betas", (0.9, 0.999))),
                                 weight_decay=t.get("weight_decay", 1e-4))
    raise ValueError(f"Unknown optimizer {name}")


def build_scheduler(optimizer, cfg):
    t = cfg["train"]
    sched = t.get("scheduler", "cosine").lower()
    total = t.get("total_iters", 300000)
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total, eta_min=t.get("eta_min", 1e-7))
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t.get("step_size", 100000),
            gamma=t.get("gamma", 0.5))
    if sched == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=t.get("milestones", [150000, 250000]),
            gamma=t.get("gamma", 0.5))
    return None


class Trainer:
    def __init__(self, model, criterion, train_loader, val_loaders, cfg,
                 device: str = "cuda", work_dir: str = "experiments/run") -> None:
        self.cfg = cfg
        self.device = device
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.join(work_dir, "checkpoints"), exist_ok=True)

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders

        t = cfg["train"]
        self.total_iters = t.get("total_iters", 300000)
        self.val_interval = t.get("val_interval", 5000)
        self.log_interval = t.get("log_interval", 100)
        self.save_interval = t.get("save_interval", 5000)
        self.grad_clip = t.get("grad_clip", 0.0)
        self.val_set_for_best = t.get("val_set_for_best", None)

        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.use_amp = t.get("amp", True) and device.startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.ema = EMA(self.model, t.get("ema_decay", 0.999)) if t.get("ema", False) else None
        self.evaluator = Evaluator(
            scale=int(cfg.get("scale", 2)),
            test_y_channel=cfg.get("eval", {}).get("test_y_channel", True),
            compute_niqe=cfg.get("eval", {}).get("compute_niqe", False),
            device=device,
        )

        self.logger = get_logger("adn", os.path.join(work_dir, "train.log"))
        self.tb = get_tensorboard_writer(os.path.join(work_dir, "tb"))
        self.tracker = MetricTracker()

        self.step = 0
        self.best_metric = -1.0

    # ------------------------------------------------------------------ #
    def _infinite_loader(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def resume(self, path: str) -> None:
        ckpt = load_checkpoint(path, self.model, self.optimizer,
                               self.scheduler, self.scaler,
                               map_location=self.device, strict=False)
        self.step = ckpt.get("step", 0)
        self.best_metric = ckpt.get("best_metric") or -1.0
        if self.ema is not None and "extra" in ckpt and "ema" in ckpt["extra"]:
            self.ema.shadow.load_state_dict(ckpt["extra"]["ema"])
        self.logger.info(f"Resumed from {path} at step {self.step}")

    def train(self) -> None:
        self.logger.info(f"Start training for {self.total_iters} iters "
                         f"(scale x{self.cfg.get('scale')}).")
        self.model.train()
        loader = self._infinite_loader()
        t0 = time.time()

        while self.step < self.total_iters:
            batch = next(loader)
            hr = batch["hr"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(hr)
                loss_dict = self.criterion(outputs, hr)
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

            self.step += 1
            self.tracker.update_dict({k: float(v) for k, v in loss_dict["logs"].items()})

            if self.step % self.log_interval == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                speed = self.log_interval / (time.time() - t0 + 1e-9)
                self.logger.info(
                    f"iter {self.step}/{self.total_iters} | lr {lr_now:.2e} | "
                    f"{self.tracker} | {speed:.1f} it/s")
                if self.tb:
                    for k, v in self.tracker.result().items():
                        self.tb.add_scalar(k, v, self.step)
                    self.tb.add_scalar("lr", lr_now, self.step)
                self.tracker.reset()
                t0 = time.time()

            if self.step % self.val_interval == 0:
                self.validate()
                self.model.train()

            if self.step % self.save_interval == 0:
                self.save("latest.pth")

        self.save("final.pth")
        self.logger.info("Training complete.")

    @torch.no_grad()
    def validate(self) -> None:
        eval_model = self.ema.shadow if self.ema is not None else self.model
        summary = {}
        for name, loader in self.val_loaders.items():
            res = self.evaluator.evaluate(eval_model, loader, name=name)
            summary[name] = res
            self.logger.info(f"[val:{name}] PSNR {res['psnr']:.4f} "
                             f"SSIM {res['ssim']:.4f} NIQE {res['niqe']:.4f}")
            if self.tb:
                self.tb.add_scalar(f"val/{name}/psnr", res["psnr"], self.step)
                self.tb.add_scalar(f"val/{name}/ssim", res["ssim"], self.step)

        best_key = self.val_set_for_best or next(iter(summary))
        metric = summary[best_key]["psnr"]
        if metric > self.best_metric:
            self.best_metric = metric
            self.save("best.pth")
            self.logger.info(f"New best PSNR {metric:.4f} on {best_key}.")

    def save(self, filename: str) -> None:
        path = os.path.join(self.work_dir, "checkpoints", filename)
        extra = {"ema": self.ema.shadow.state_dict()} if self.ema is not None else None
        save_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.scaler,
            step=self.step, best_metric=self.best_metric,
            config=self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg),
            extra=extra,
        )
