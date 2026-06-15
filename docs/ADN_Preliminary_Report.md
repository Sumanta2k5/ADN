# Preliminary Report: Adaptive Downscaling Network (ADN)

## 1. Objective

This preliminary experiment studies an Adaptive Downscaling Network (ADN) for image downscaling and reconstruction-based evaluation. The model takes a high-resolution image, generates a learned low-resolution image, then reconstructs it using bicubic upscaling and compares the reconstruction with the original HR image using PSNR and SSIM.

This is a pilot run for demonstration and progress review. It is not the final publication-level training run.

## 2. Paper Reference and Dataset Requirement

The ADN paper states in the Experimental Setup / Datasets section that the model is trained and evaluated using the DIV2K dataset. It describes DIV2K as 900 high-resolution images divided into:

| Split | Number of Images |
|---|---:|
| Training | 800 |
| Validation | 50 |
| Test | 50 |

The paper also mentions training with randomly cropped HR patches of sizes 128 x 128, 256 x 256, and 512 x 512, with random horizontal and vertical flips.

For benchmark evaluation, the paper uses:

| Benchmark Dataset | Number of Images |
|---|---:|
| Set5 | 5 |
| Set14 | 14 |
| BSD100 | 100 |
| Urban100 | 100 |
| DIV2K validation | 50 |

## 3. Dataset Placement Used in Code

The code expects the following folder structure:

```text
C:\Users\suman\Documents\Codex\ADN-run\ADN\
  datasets\
    DIV2K\
      DIV2K_train_HR\
      DIV2K_valid_HR\
    benchmark\
      Set5\HR\
      Set14\HR\
      BSD100\HR\
      Urban100\HR\
```

Only HR images are required for this ADN code. The LR images included in some benchmark downloads are ignored because the model generates its own LR output from the HR input.

## 4. Model Architecture Followed

The implementation follows the main ADN pipeline described in the paper:

```text
HR image
-> Feature Extractor
-> ARM / CBAM Attention
-> Kernel Generation Module (KGM)
-> Offset Estimation Module (OEM)
-> Adaptive Resampling Layer
-> LR image
-> Bicubic reconstruction
-> MSE + SSIM loss against HR
```

Important implemented components:

| Paper Component | Code Module |
|---|---|
| Feature extractor | `adn/models/feature_extractor.py` |
| ARM / CBAM attention | `adn/models/attention.py` |
| Kernel Generation Module | `adn/models/kernel_offset.py` |
| Offset Estimation Module | `adn/models/kernel_offset.py` |
| Adaptive resampling | `adn/models/resampler.py` |
| Hybrid MSE + SSIM loss | `adn/losses/losses.py` |
| Bicubic reconstructor | `adn/models/reconstructor.py` |

Some details are implementation assumptions because the paper does not specify every training and architecture parameter.

## 5. Commands Used

### 5.1 Create Dataset Folders

```powershell
cd "C:\Users\suman\Documents\Codex\ADN-run\ADN"

New-Item -ItemType Directory -Force -Path ".\datasets\DIV2K\DIV2K_train_HR"
New-Item -ItemType Directory -Force -Path ".\datasets\DIV2K\DIV2K_valid_HR"
New-Item -ItemType Directory -Force -Path ".\datasets\benchmark\Set5\HR"
New-Item -ItemType Directory -Force -Path ".\datasets\benchmark\Set14\HR"
New-Item -ItemType Directory -Force -Path ".\datasets\benchmark\BSD100\HR"
New-Item -ItemType Directory -Force -Path ".\datasets\benchmark\Urban100\HR"
```

### 5.2 Pilot Training Command

```powershell
cd "C:\Users\suman\Documents\Codex\ADN-run\ADN"

.\.venv\Scripts\python.exe -m scripts.train --config configs/adn_x4.yaml --work-dir experiments/adn_x4_pilot --opts train.total_iters=1000 train.log_interval=50 train.val_interval=500 train.save_interval=500 data.batch_size=4 data.patch_size=128 data.num_workers=2 data.val_max_images=5 train.amp=false
```

### 5.3 Track Training Progress

```powershell
Get-Content .\experiments\adn_x4_pilot\train.log -Wait
```

### 5.4 Generate Final Test Images and Metrics

```powershell
.\.venv\Scripts\python.exe -m scripts.test --config configs/adn_x4.yaml --checkpoint experiments/adn_x4_pilot/checkpoints/best.pth --save-dir results/adn_x4_pilot --opts data.val_max_images=5
```

### 5.5 View Metrics

```powershell
Get-Content .\results\adn_x4_pilot\metrics.json
```

## 6. Preliminary Training Configuration

| Setting | Value |
|---|---:|
| Scale factor | x4 |
| Device | CPU |
| Real training images | 800 DIV2K HR images |
| Virtual training samples shown by code | 16000 because repeat = 20 |
| Batch size | 4 |
| Total iterations | 1000 |
| Total HR patches used | 4000 patches |
| Patch size | 128 x 128 |
| Validation limit | 5 images per dataset |
| Loss | 0.55 MSE + 0.45 SSIM |
| Optimizer | Adam |
| Learning rate | 0.0002 with cosine schedule |

The training log showed:

```text
Device: cpu
Train images: 16000
Model params: 1.864 M
Start training for 1000 iters (scale x4)
Training complete
```

## 7. Preliminary Results From This Run

These values are from:

```text
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\metrics.json
```

| Dataset | Images Used | PSNR | SSIM | NIQE |
|---|---:|---:|---:|---:|
| Set5 | 5 | 28.5357 | 0.8310 | 8.8941 |
| Set14 | 5 | 23.6859 | 0.5972 | 10.4907 |
| BSD100 | 5 | 25.8428 | 0.7160 | NaN |
| Urban100 | 5 | 23.0494 | 0.7051 | 15.7026 |
| DIV2K validation | 5 | 30.9613 | 0.8404 | 4.2420 |

## 8. Comparison With Known Downscaling Methods

The baseline values below are x4 PSNR / SSIM values reported in the ADN paper. The final row is the preliminary result from this pilot run.

| Method | Set5 | Set14 | BSD100 | Urban100 | DIV2K Val |
|---|---:|---:|---:|---:|---:|
| Bicubic | 25.94 / 0.7714 | 23.10 / 0.6641 | 23.32 / 0.6403 | 20.45 / 0.6341 | 26.01 / 0.7694 |
| DPID | 24.72 / 0.7368 | 23.36 / 0.6436 | 24.00 / 0.6256 | 20.98 / 0.6159 | 26.20 / 0.7551 |
| IDCL | 24.62 / 0.7358 | 23.26 / 0.6438 | 23.91 / 0.6244 | 20.83 / 0.6156 | 26.14 / 0.7552 |
| ADN Paper | 26.87 / 0.7814 | 24.45 / 0.6808 | 24.70 / 0.6521 | 21.84 / 0.6488 | 27.36 / 0.7923 |
| ADN Pilot Run | 28.54 / 0.8310 | 23.69 / 0.5972 | 25.84 / 0.7160 | 23.05 / 0.7051 | 30.96 / 0.8404 |

Note: The pilot run used only 5 validation/test images per dataset, so these preliminary results are not final publication-level numbers. The full experiment should evaluate on the complete Set5, Set14, BSD100, Urban100, and DIV2K validation sets.

## 9. Example Output Images

The test script generated LR and SR output images in:

```text
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\
```

Example output sheet:

![Example ADN outputs](adn_preliminary_report_assets/example_outputs_sheet.png)

Individual examples:

| Dataset | LR Output | SR Output |
|---|---|---|
| Set5 | `adn_preliminary_report_assets/set5_example_LR.png` | `adn_preliminary_report_assets/set5_example_SR.png` |
| Urban100 | `adn_preliminary_report_assets/urban100_example_LR.png` | `adn_preliminary_report_assets/urban100_example_SR.png` |

## 10. Files Generated

Important files from the pilot run:

```text
C:\Users\suman\Documents\Codex\ADN-run\ADN\experiments\adn_x4_pilot\train.log
C:\Users\suman\Documents\Codex\ADN-run\ADN\experiments\adn_x4_pilot\checkpoints\best.pth
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\metrics.json
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\Set5\
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\Set14\
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\BSD100\
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\Urban100\
C:\Users\suman\Documents\Codex\ADN-run\ADN\results\adn_x4_pilot\DIV2K\
```

## 11. Conclusion

A preliminary ADN x4 training run was completed successfully on CPU for 1000 iterations. The model was trained using DIV2K HR patches and evaluated on Set5, Set14, BSD100, Urban100, and DIV2K validation with a 5-image limit per dataset. The run generated both quantitative metrics and LR/SR output images. For publication-quality results, the next stage is longer training on the full DIV2K training set and evaluation on the complete benchmark datasets.
