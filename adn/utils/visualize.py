"""Visualization helpers: comparison grids, kernel heatmaps, offset fields."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return (t.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)


def save_comparison_grid(images: Dict[str, torch.Tensor], out_path: str,
                         titles: Optional[List[str]] = None, dpi: int = 200) -> None:
    """Save a row of labelled images (e.g. HR | Bicubic | ADN) with PSNR labels."""
    import matplotlib.pyplot as plt

    keys = list(images.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.imshow(tensor_to_np(images[k]))
        ax.set_title(k, fontsize=10)
        ax.axis("off")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_kernel_heatmap(kernels: torch.Tensor, out_path: str, stride: int = 32) -> None:
    """Visualize the spatial entropy / sharpness of predicted kernels.

    ``kernels``: (k*k, H, W). We render the per-pixel kernel entropy (low entropy
    = sharp, content-adaptive kernel) as a heatmap, as discussed in the paper's
    kernel-heatmap analysis.
    """
    import matplotlib.pyplot as plt

    k2, h, w = kernels.shape
    probs = kernels.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(0)            # (H, W)
    entropy = entropy / np.log(k2)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(entropy.cpu().numpy(), cmap="viridis")
    ax.set_title("Kernel entropy (lower = sharper)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_offset_field(offset_x: torch.Tensor, offset_y: torch.Tensor,
                      out_path: str, step: int = 8, tap: int = 0) -> None:
    """Quiver plot of the predicted offset field for a chosen kernel tap."""
    import matplotlib.pyplot as plt

    dx = offset_x[tap].cpu().numpy()
    dy = offset_y[tap].cpu().numpy()
    h, w = dx.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(xs, ys, dx[::step, ::step], -dy[::step, ::step],
              color="crimson", scale=None, angles="xy")
    ax.set_title(f"Offset field (tap {tap})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
