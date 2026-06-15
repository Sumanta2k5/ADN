"""Image augmentation / cropping utilities (numpy-based, fast)."""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8/float numpy in [0,255] -> CHW float tensor in [0,1]."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
    if img.ndim == 2:
        img = img[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def random_crop(img: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w = img.shape[:2]
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return img[top:top + patch_size, left:left + patch_size]


def paired_random_crop(imgs: List[np.ndarray], patch_size: int,
                       scale: int = 1) -> List[np.ndarray]:
    """Crop a list of aligned images; ``scale`` lets HR/LR pairs stay aligned."""
    ref = imgs[0]
    h, w = ref.shape[:2]
    top = random.randint(0, max(0, h - patch_size))
    left = random.randint(0, max(0, w - patch_size))
    out = []
    for k, im in enumerate(imgs):
        s = 1 if k == 0 else scale
        out.append(im[top * s:(top + patch_size) * s,
                      left * s:(left + patch_size) * s])
    return out


def augment(img: np.ndarray, hflip: bool = True, rot: bool = True) -> np.ndarray:
    """Random horizontal/vertical flips and 90-deg rotation."""
    do_hflip = hflip and random.random() < 0.5
    do_vflip = rot and random.random() < 0.5
    do_rot90 = rot and random.random() < 0.5
    if do_hflip:
        img = img[:, ::-1, :]
    if do_vflip:
        img = img[::-1, :, :]
    if do_rot90:
        img = img.transpose(1, 0, 2)
    return np.ascontiguousarray(img)


def modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    """Crop image so H and W are divisible by ``scale``."""
    h, w = img.shape[:2]
    h = h - h % scale
    w = w - w % scale
    return img[:h, :w]


def ensure_divisible(img: np.ndarray, factor: int) -> np.ndarray:
    """Crop so dims are divisible by ``factor`` (e.g. feature_downscale*scale)."""
    h, w = img.shape[:2]
    h = h - h % factor
    w = w - w % factor
    return img[:h, :w]
