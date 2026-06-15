#!/usr/bin/env bash
# Reproduce all ADN experiments end-to-end: train x2/x4/x8, evaluate on all
# benchmarks, run ablations, and regenerate tables + figures.
#
# Usage:
#   bash scripts/reproduce_all.sh [DATA_ROOT] [DEVICE]
#
# Requires datasets prepared via scripts/download_datasets.py.
set -e

DATA_ROOT=${1:-datasets}
DEVICE=${2:-cuda}
PY="python -m"

echo "==> Verifying datasets under ${DATA_ROOT}"
$PY scripts.download_datasets --root "${DATA_ROOT}"

# ----------------------------------------------------------------------------
# 1. Main models: x2, x4, x8
# ----------------------------------------------------------------------------
for SCALE in 2 4 8; do
  CFG=configs/adn_x${SCALE}.yaml
  echo "==> Training ADN x${SCALE}"
  $PY scripts.train --config ${CFG} --device ${DEVICE}

  echo "==> Evaluating ADN x${SCALE}"
  $PY scripts.test --config ${CFG} \
      --checkpoint experiments/adn_x${SCALE}/checkpoints/best.pth \
      --save-dir results/adn_x${SCALE} --device ${DEVICE}
done

# ----------------------------------------------------------------------------
# 2. Ablations (x2): w/o CBAM, w/o ResNet, patch size, kernel depth
# ----------------------------------------------------------------------------
for ABL in wo_cbam wo_resnet patch128 patch512 kdepth3 kdepth9; do
  CFG=configs/ablation/${ABL}.yaml
  echo "==> Ablation: ${ABL}"
  $PY scripts.train --config ${CFG} --device ${DEVICE}
  $PY scripts.test --config ${CFG} \
      --checkpoint experiments/ablation_${ABL}/checkpoints/best.pth \
      --save-dir results/ablation_${ABL} --device ${DEVICE}
done

# ----------------------------------------------------------------------------
# 3. Tables and figures
# ----------------------------------------------------------------------------
echo "==> Generating tables"
$PY scripts.make_tables --results-dir results --out tables

echo "==> Generating qualitative figures (x4 on Urban100)"
$PY scripts.make_figures --config configs/adn_x4.yaml \
    --checkpoint experiments/adn_x4/checkpoints/best.pth \
    --dataset "${DATA_ROOT}/benchmark/Urban100/HR" --num 5 --out figures/adn_x4

echo "==> Done. See tables/ and figures/."
