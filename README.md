# ADN: Adaptive Image Downscaling Network

A complete, conference-quality **PyTorch reproduction** of
**"ADN: Adaptive Image Downscaling Network"** (Pise & Ghosh).

ADN is a learned, content-adaptive image **downscaler** for the
*downscale-for-upscale* task. It encodes the HR image, refines features with
CBAM attention (ARM), predicts per-pixel **resampling kernels** (KGM) and
**deformable sampling offsets** (OEM), and performs differentiable bilinear
resampling to produce a perceptually optimized LR image. It is trained with a
hybrid **MSE + SSIM** objective so the LR reconstructs the HR well after
upscaling.

> This is an **unofficial** reproduction. The original PDF's §VII–XII are IEEE
> LaTeX template boilerplate; only the genuine method (§III–V) is implemented.
> See [Assumptions](#assumptions) for choices made where the paper is
> underspecified.

---

## Highlights

- Faithful implementation of every paper equation (channel/spatial attention,
  HR projection, deformable taps, kernel mixing, hybrid loss).
- **Adaptive Resampling Layer** with vectorized, differentiable `grid_sample`.
- MATLAB-compatible **bicubic `imresize`** for benchmark-accurate PSNR/SSIM.
- Full training engine: AMP, EMA, cosine schedule, TensorBoard, resumable ckpts.
- Benchmark evaluation on **Set5 / Set14 / BSD100 / Urban100 / DIV2K** with
  **PSNR / SSIM** (Y channel, scale-shaved) and **NIQE**.
- Config-driven **ablation framework** (w/o CBAM, w/o ResNet, patch size,
  kernel depth) reproducing Table V.
- One-command reproduction + automatic **table & figure generation**.

---

## Project structure

```
ADN/
├── adn/
│   ├── models/        # feature extractor, ARM/CBAM, KGM, OEM, resampler, ADN, reconstructor
│   ├── losses/        # SSIM engine + hybrid MSE+SSIM loss (Eq. 4)
│   ├── metrics/       # PSNR, SSIM (SR convention), NIQE
│   ├── data/          # datasets, transforms, preparation
│   ├── engine/        # Trainer, Evaluator
│   └── utils/         # config, logging, checkpoint, seed, imresize, color, visualize
├── configs/           # base + adn_x2/x4/x8 + ablation/*  (+ adn_x4_sr)
├── scripts/           # train, test, infer, download_datasets, make_tables, make_figures, reproduce_all.sh
├── tests/             # pytest unit tests (resampler, model, metrics)
├── requirements.txt, setup.py, LICENSE, README.md
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install -e .                                        # optional: console entry points
```

PyTorch ≥ 2.0 with CUDA is recommended. `pyiqa` is optional (enables NIQE).

---

## Datasets

```bash
# Download DIV2K and print benchmark instructions
python -m scripts.download_datasets --root datasets --download

# (optional) tile DIV2K HR into sub-images for faster I/O
python -m scripts.download_datasets --root datasets --prepare-subimages
```

Expected layout:

```
datasets/
  DIV2K/DIV2K_train_HR/*.png
  DIV2K/DIV2K_valid_HR/*.png
  benchmark/{Set5,Set14,BSD100,Urban100}/HR/*.png
```

---

## Training

```bash
python -m scripts.train --config configs/adn_x2.yaml
python -m scripts.train --config configs/adn_x4.yaml
python -m scripts.train --config configs/adn_x8.yaml

# override any field on the CLI
python -m scripts.train --config configs/adn_x4.yaml --opts train.total_iters=100000 data.batch_size=8
```

Outputs (logs, TensorBoard, checkpoints) go to `experiments/<name>/`.
Watch training with `tensorboard --logdir experiments`.

---

## Evaluation

```bash
python -m scripts.test --config configs/adn_x4.yaml \
    --checkpoint experiments/adn_x4/checkpoints/best.pth \
    --save-dir results/adn_x4
```

Writes `results/adn_x4/metrics.json` (+ per-image JSON + saved LR/SR images).
PSNR/SSIM are computed on the Y channel with `crop_border = scale`, matching the
SR literature; NIQE is computed on the LR output.

---

## Inference on your own images

```bash
python -m scripts.infer --config configs/adn_x4.yaml \
    --checkpoint experiments/adn_x4/checkpoints/best.pth \
    --input path/to/images --output infer_out --save-recon
```

---

## Reproduce everything (tables + figures)

```bash
bash scripts/reproduce_all.sh datasets cuda
```

This trains x2/x4/x8 + all ablations, evaluates on every benchmark, then runs:

```bash
python -m scripts.make_tables  --results-dir results --out tables
python -m scripts.make_figures --config configs/adn_x4.yaml \
    --checkpoint experiments/adn_x4/checkpoints/best.pth \
    --dataset datasets/benchmark/Urban100/HR --num 5 --out figures/adn_x4
```

- `tables/` → Markdown + LaTeX versions of the benchmark and ablation tables.
- `figures/` → HR/Bicubic/ADN comparison grids (Figs. 3–6), kernel-entropy
  heatmaps (Sec. V), and offset quiver fields.

---

## Method overview

```
HR ──FeatureExtractor──▶ F (H/8) ──ARM/CBAM──▶ F_r ──resize→H/s──▶ ┌ KGM ▶ kernels K
                                                                   └ OEM ▶ offsets ΔX,ΔY
                              AdaptiveResampling(HR, K, ΔX, ΔY) ──▶ LR ──recon──▶ HR̂
```

| Component | File | Paper ref |
|---|---|---|
| Feature extractor | [adn/models/feature_extractor.py](adn/models/feature_extractor.py) | §III-B |
| ARM (CBAM) | [adn/models/attention.py](adn/models/attention.py) | §III-B, Eqs. A_c/A_s |
| KGM / OEM | [adn/models/kernel_offset.py](adn/models/kernel_offset.py) | §III-B |
| Adaptive Resampling | [adn/models/resampler.py](adn/models/resampler.py) | Eqs. 1–3 |
| Hybrid loss | [adn/losses/losses.py](adn/losses/losses.py) | Eq. 4 |
| ADN model | [adn/models/adn.py](adn/models/adn.py) | §III-A |

---

## Assumptions

Where the paper is missing/ambiguous, we follow CVPR/ICCV/ECCV SR practice
(all overridable via config):

| Item | Paper | Default here |
|---|---|---|
| Kernel size `m,n` | unspecified | `kernel_size=3` |
| Feature channels `C` | unspecified | `64` |
| Tap centering | "`i − m/2`" (asymmetric) | symmetric `(k−1)/2` (`paper_centering=false`) |
| Optimizer | "Adam + LR schedule" | Adam, lr `2e-4`, cosine, 300k iters |
| Batch / patch | patches 128/256/512 | bs 16, HR patch 192 (x2) |
| Reconstruction for loss | "bicubic upscaling" | differentiable bicubic (or `edsr` head) |
| LR regularization | not mentioned | optional guidance loss (weight 0 → faithful) |
| NIQE | reported | via `pyiqa` (skipped w/ warning if absent) |

The paper's Table V contains internal inconsistencies/placeholders; we implement
the method honestly and regenerate all numbers from real runs rather than
hard-coding the printed values.

---

## Testing

```bash
pytest -q
```

Covers resampler shape/brightness/gradient correctness, model forward/backward at
x2/x4 (incl. ablation variants), and metric/loss sanity.

---

## Citation

```bibtex
@article{pise_ghosh_adn,
  title   = {ADN: Adaptive Image Downscaling Network},
  author  = {Pise, Piyush Narhari and Ghosh, Sanjay},
  journal = {Preprint},
  year    = {2025}
}
```

Builds on CBAM (Woo et al., ECCV 2018) and content-adaptive resampling
(Sun & Chen, TIP 2020). Released under the MIT License.
