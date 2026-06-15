"""Download and organize the datasets used in the paper.

Datasets:
  * DIV2K (train HR, valid HR)          -- training & validation
  * Set5, Set14, BSD100, Urban100       -- standard SR test benchmarks

This script prints official URLs and, when ``--download`` is given, fetches the
DIV2K archives directly. The classic test sets are distributed in various
mirrors; we point to commonly used ones and verify after extraction.

Usage:
    python -m scripts.download_datasets --root datasets --download
    python -m scripts.download_datasets --root datasets --prepare-subimages
"""
from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile

from adn.data.prepare import extract_subimages, verify_datasets

# Chunk size for streaming downloads (8 MiB).
_CHUNK = 8 * 1024 * 1024

DIV2K_URLS = {
    "DIV2K_train_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

BENCHMARK_INFO = """
Standard SR test sets (Set5, Set14, BSD100, Urban100) are available from common
mirrors, e.g. the BasicSR / RCAN benchmark bundle:

  https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md
  https://cv.snu.ac.kr/research/EDSR/benchmark.tar   (Set5/Set14/B100/Urban100 HR)

After downloading, arrange them as:

  datasets/
    DIV2K/DIV2K_train_HR/*.png
    DIV2K/DIV2K_valid_HR/*.png
    benchmark/Set5/HR/*.png
    benchmark/Set14/HR/*.png
    benchmark/BSD100/HR/*.png
    benchmark/Urban100/HR/*.png
"""


def _download(url: str, dst: str, retries: int = 5) -> None:
    """Stream ``url`` to ``dst`` with resume support and integrity checks.

    The previous implementation used ``urllib.request.urlretrieve`` which, on a
    dropped connection (common with the ETH DIV2K mirror), left a truncated file
    behind that later poisoned extraction with ``BadZipFile``. This version:

      * resumes via HTTP Range requests when a ``.part`` file exists,
      * validates the final size against the server's Content-Length,
      * only renames ``.part`` -> ``dst`` once the download is complete,
      * retries a configurable number of times.
    """
    if os.path.exists(dst):
        print(f"[skip] {dst} already exists.")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"

    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
            print(f"[resume] {dst} from {have/1e6:.1f} MB (attempt {attempt}/{retries})")
        else:
            print(f"[get ] {url} (attempt {attempt}/{retries})")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Total expected size = bytes already on disk + remaining.
                remaining = resp.length or 0
                total = have + remaining if remaining else None
                mode = "ab" if (have and resp.status == 206) else "wb"
                if mode == "wb":
                    have = 0  # server ignored Range; restart cleanly
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if total:
                            pct = 100.0 * have / total
                            print(f"\r  {have/1e6:8.1f} / {total/1e6:.1f} MB "
                                  f"({pct:5.1f}%)", end="", flush=True)
                if total:
                    print()
            # Verify completeness if we know the expected size.
            if total and os.path.getsize(part) < total:
                raise IOError(
                    f"incomplete: {os.path.getsize(part)} < {total} bytes")
            os.replace(part, dst)
            print(f"[done] {dst}")
            return
        except Exception as exc:  # noqa: BLE001 - report and retry
            print(f"\n[warn] download failed: {exc}")
            if attempt == retries:
                if os.path.exists(part):
                    print(f"[keep] partial file kept for resume: {part}")
                raise


def _extract(zip_path: str, out_dir: str) -> None:
    print(f"[unzip] {zip_path} -> {out_dir}")
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(
            f"{zip_path} is not a valid zip (corrupt/partial). "
            f"Delete it and re-run --download.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def _make_dummy(root: str, per_set: int = 8, size: int = 256) -> None:
    """Synthesize a tiny dataset so the pipeline can run end-to-end offline.

    Generates colorful procedural HR images (gradients + sinusoids + noise) into
    the exact folder layout the configs expect. Intended for smoke-testing
    training/eval on CPU, not for reproducing paper numbers.
    """
    import numpy as np
    from PIL import Image

    folders = {
        os.path.join(root, "DIV2K", "DIV2K_train_HR"): per_set,
        os.path.join(root, "DIV2K", "DIV2K_valid_HR"): max(2, per_set // 4),
        os.path.join(root, "benchmark", "Set5", "HR"): 5,
        os.path.join(root, "benchmark", "Set14", "HR"): 5,
        os.path.join(root, "benchmark", "BSD100", "HR"): 5,
        os.path.join(root, "benchmark", "Urban100", "HR"): 5,
    }
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for folder, count in folders.items():
        os.makedirs(folder, exist_ok=True)
        for i in range(count):
            fx, fy = rng.uniform(2, 12, size=2)
            phase = rng.uniform(0, 6.28, size=3)
            r = 128 + 110 * np.sin(2 * np.pi * fx * xx / size + phase[0])
            g = 128 + 110 * np.sin(2 * np.pi * fy * yy / size + phase[1])
            b = 128 + 110 * np.sin(2 * np.pi * (xx + yy) / size + phase[2])
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 8, img.shape)
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(folder, f"dummy_{i:03d}.png"))
        print(f"[dummy] {count:3d} images -> {folder}")


def main():
    p = argparse.ArgumentParser(description="Download ADN datasets")
    p.add_argument("--root", default="datasets")
    p.add_argument("--download", action="store_true", help="Download DIV2K archives.")
    p.add_argument("--prepare-subimages", action="store_true",
                   help="Tile DIV2K train HR into sub-images for faster I/O.")
    p.add_argument("--dummy", type=int, default=0, metavar="N",
                   help="Generate N synthetic HR train images (+ small val/benchmark) "
                        "for offline smoke-testing instead of downloading.")
    args = p.parse_args()

    div2k_dir = os.path.join(args.root, "DIV2K")
    os.makedirs(div2k_dir, exist_ok=True)

    if args.dummy:
        _make_dummy(args.root, per_set=args.dummy)

    if args.download:
        for fname, url in DIV2K_URLS.items():
            zip_path = os.path.join(div2k_dir, fname)
            _download(url, zip_path)
            _extract(zip_path, div2k_dir)

    print(BENCHMARK_INFO)

    if args.prepare_subimages:
        src = os.path.join(div2k_dir, "DIV2K_train_HR")
        dst = os.path.join(div2k_dir, "DIV2K_train_HR_sub")
        if os.path.isdir(src):
            extract_subimages(src, dst, crop_size=480, step=240)
        else:
            print(f"[warn] {src} not found; download DIV2K first.")

    verify_datasets({
        "DIV2K_train": os.path.join(div2k_dir, "DIV2K_train_HR"),
        "DIV2K_valid": os.path.join(div2k_dir, "DIV2K_valid_HR"),
        "Set5": os.path.join(args.root, "benchmark", "Set5", "HR"),
        "Set14": os.path.join(args.root, "benchmark", "Set14", "HR"),
        "BSD100": os.path.join(args.root, "benchmark", "BSD100", "HR"),
        "Urban100": os.path.join(args.root, "benchmark", "Urban100", "HR"),
    })


if __name__ == "__main__":
    main()
