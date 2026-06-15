"""Logging utilities: console + file logger, TensorBoard, and metric averaging."""
from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from typing import Dict, Optional

_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(
    name: str = "adn",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger (idempotent per name)."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


class AverageMeter:
    """Running average for a single scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class MetricTracker:
    """Tracks multiple named running averages."""

    def __init__(self) -> None:
        self._meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, name: str, value: float, n: int = 1) -> None:
        self._meters[name].update(value, n)

    def update_dict(self, values: Dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self.update(k, v, n)

    def avg(self, name: str) -> float:
        return self._meters[name].avg

    def result(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}

    def reset(self) -> None:
        for m in self._meters.values():
            m.reset()

    def __str__(self) -> str:
        return ", ".join(f"{k}: {v:.4f}" for k, v in self.result().items())


def get_tensorboard_writer(log_dir: str):
    """Return a SummaryWriter, or None if TensorBoard is unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # pragma: no cover
        return None
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)
