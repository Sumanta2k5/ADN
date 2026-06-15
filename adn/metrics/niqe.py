"""NIQE (no-reference) perceptual quality metric.

NIQE requires the standard pristine MVG model parameters. Rather than vendor a
binary model file, we delegate to ``pyiqa`` (which ships the official model) when
available. If ``pyiqa`` is not installed, NIQE is reported as NaN and skipped
with a one-time warning, so the rest of the evaluation still runs.

Lower NIQE = better perceptual quality.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch

_NIQE_MODEL = None
_WARNED = False

# Minimum side length (px) for which NIQE is meaningful/stable. NIQE pools
# statistics over 96x96 blocks after Gaussian filtering, so smaller images are
# skipped rather than crashing pyiqa.
_MIN_NIQE_SIZE = 96


def niqe_available() -> bool:
    try:
        import pyiqa  # noqa: F401
        return True
    except Exception:
        return False


def _get_model(device: str = "cpu"):
    global _NIQE_MODEL
    if _NIQE_MODEL is None:
        import pyiqa
        _NIQE_MODEL = pyiqa.create_metric("niqe", device=device)
    return _NIQE_MODEL


def calculate_niqe(img: torch.Tensor, device: Optional[str] = None) -> float:
    """Compute NIQE for an RGB image tensor (C,H,W) or (B,C,H,W) in [0,1]."""
    global _WARNED
    if not niqe_available():
        if not _WARNED:
            warnings.warn("pyiqa not installed; NIQE will be reported as NaN. "
                          "Install with `pip install pyiqa` to enable it.")
            _WARNED = True
        return float("nan")

    if img.dim() == 3:
        img = img.unsqueeze(0)
    dev = device or ("cuda" if img.is_cuda else "cpu")

    # NIQE convolves with a 7x7 Gaussian and pools over blocks; very small
    # images (e.g. x4/x8 LR outputs of small HR patches) collapse to zero-size
    # tensors inside pyiqa and raise. Skip them gracefully instead of crashing.
    h, w = img.shape[-2:]
    if min(h, w) < _MIN_NIQE_SIZE:
        if not _WARNED:
            warnings.warn(
                f"NIQE skipped for image smaller than {_MIN_NIQE_SIZE}px "
                f"(got {h}x{w}); reported as NaN.")
            _WARNED = True
        return float("nan")

    model = _get_model(dev)
    try:
        with torch.no_grad():
            score = model(img.clamp(0, 1).to(dev))
    except RuntimeError as exc:
        if not _WARNED:
            warnings.warn(f"NIQE failed ({exc}); reported as NaN.")
            _WARNED = True
        return float("nan")
    return float(score.mean().item())
