"""Dataset preparation utilities.

* Optionally crops DIV2K HR training images into sub-images for faster I/O
  (a common SR practice; e.g. 480x480 with stride 240).
* Verifies dataset folders exist and reports image counts.

Run via ``scripts/download_datasets.py`` (downloading) and this module for
post-processing.
"""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(root: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
    return sorted(files)


def extract_subimages(src: str, dst: str, crop_size: int = 480,
                      step: int = 240, thresh: int = 0) -> None:
    """Tile each image in ``src`` into ``crop_size`` patches into ``dst``."""
    os.makedirs(dst, exist_ok=True)
    paths = list_images(src)
    print(f"[prepare] {len(paths)} images -> sub-images in {dst}")
    for path in tqdm(paths):
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        name = os.path.splitext(os.path.basename(path))[0]
        h_space = list(range(0, max(1, h - crop_size + 1), step))
        w_space = list(range(0, max(1, w - crop_size + 1), step))
        if h_space[-1] + crop_size < h:
            h_space.append(h - crop_size)
        if w_space[-1] + crop_size < w:
            w_space.append(w - crop_size)
        idx = 0
        for top in h_space:
            for left in w_space:
                patch = img[top:top + crop_size, left:left + crop_size]
                if patch.shape[0] < crop_size or patch.shape[1] < crop_size:
                    continue
                idx += 1
                out = os.path.join(dst, f"{name}_s{idx:03d}.png")
                Image.fromarray(patch).save(out)


def verify_datasets(roots: dict) -> None:
    print("=" * 50)
    for name, root in roots.items():
        if root and os.path.isdir(root):
            n = len(list_images(root))
            status = "OK " if n > 0 else "EMPTY"
            print(f"[{status}] {name:12s} {n:5d} images  ({root})")
        else:
            print(f"[MISS] {name:12s}  (missing: {root})")
    print("=" * 50)
