"""Datasets and dataloader builders.

ADN is self-supervised at the image level: it consumes an HR image, produces an
LR, and reconstructs HR for the loss. Therefore the dataset only needs HR
images. We crop training patches divisible by ``align`` (= feature_downscale *
scale) so the H/8 encoder and the H/s resampler stay grid-aligned.
"""
from __future__ import annotations

import glob
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from adn.data.transforms import augment, ensure_divisible, random_crop, to_tensor
from adn.utils.seed import seed_worker

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(root: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
        files += glob.glob(os.path.join(root, f"**/*{ext.upper()}"), recursive=True)
    return sorted(set(files))


def _read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


class HRImageDataset(Dataset):
    """Training dataset yielding augmented HR patches."""

    def __init__(self, root: str, patch_size: int = 192, scale: int = 2,
                 feature_downscale: int = 8, augment_data: bool = True,
                 repeat: int = 1, cache: bool = False) -> None:
        super().__init__()
        self.paths = _list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.patch_size = patch_size
        self.scale = scale
        self.align = max(scale, feature_downscale) if feature_downscale % scale == 0 \
            else feature_downscale * scale
        self.augment_data = augment_data
        self.repeat = max(1, repeat)
        self.cache = cache
        self._cache: Dict[int, np.ndarray] = {}
        # Make patch_size divisible by align.
        if self.patch_size % self.align != 0:
            self.patch_size -= self.patch_size % self.align

    def __len__(self) -> int:
        return len(self.paths) * self.repeat

    def _load(self, idx: int) -> np.ndarray:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        img = _read_image(self.paths[idx])
        if self.cache:
            self._cache[idx] = img
        return img

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        idx = index % len(self.paths)
        img = self._load(idx)
        patch = random_crop(img, self.patch_size)
        if self.augment_data:
            patch = augment(patch)
        hr = to_tensor(patch)
        return {"hr": hr, "path": self.paths[idx]}


class DIV2KDataset(HRImageDataset):
    """Convenience subclass for DIV2K HR folders."""


class BenchmarkDataset(Dataset):
    """Validation/test dataset yielding full HR images (mod-cropped)."""

    def __init__(self, root: str, scale: int = 2, feature_downscale: int = 8,
                 max_images: Optional[int] = None) -> None:
        super().__init__()
        self.paths = _list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.scale = scale
        self.align = feature_downscale * scale // _gcd(feature_downscale, scale)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict:
        path = self.paths[index]
        img = ensure_divisible(_read_image(path), self.align)
        hr = to_tensor(img)
        name = os.path.splitext(os.path.basename(path))[0]
        return {"hr": hr, "name": name, "path": path}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def build_dataloaders(cfg) -> Tuple[Optional[DataLoader], Dict[str, DataLoader]]:
    """Build the training loader and a dict of {name: val_loader}."""
    data = cfg["data"]
    scale = int(cfg.get("scale", 2))
    feat_down = cfg.get("model", {}).get("feature_downscale", 8)

    train_loader = None
    if data.get("train_root"):
        train_set = HRImageDataset(
            root=data["train_root"],
            patch_size=data.get("patch_size", 192),
            scale=scale,
            feature_downscale=feat_down,
            augment_data=data.get("augment", True),
            repeat=data.get("repeat", 1),
            cache=data.get("cache", False),
        )
        g = torch.Generator()
        g.manual_seed(cfg.get("seed", 42))
        train_loader = DataLoader(
            train_set,
            batch_size=data.get("batch_size", 16),
            shuffle=True,
            num_workers=data.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=data.get("num_workers", 4) > 0,
        )

    val_loaders: Dict[str, DataLoader] = {}
    for name, root in (data.get("val_roots", {}) or {}).items():
        if not root or not os.path.isdir(root) or not _list_images(root):
            warnings.warn(
                f"Validation set '{name}' skipped: no images found under {root}")
            continue
        val_set = BenchmarkDataset(
            root=root, scale=scale, feature_downscale=feat_down,
            max_images=data.get("val_max_images"),
        )
        val_loaders[name] = DataLoader(
            val_set, batch_size=1, shuffle=False,
            num_workers=data.get("num_workers", 4), pin_memory=True,
        )
    return train_loader, val_loaders
