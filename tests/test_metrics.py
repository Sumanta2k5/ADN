import torch

from adn.losses import HybridDownscaleLoss
from adn.losses.ssim import ssim
from adn.metrics import calculate_psnr, calculate_ssim


def test_psnr_identical_is_high():
    img = torch.rand(1, 3, 32, 32)
    assert calculate_psnr(img, img.clone()) > 80


def test_ssim_identical_is_one():
    img = torch.rand(1, 3, 32, 32)
    assert abs(calculate_ssim(img, img.clone()) - 1.0) < 1e-4


def test_ssim_engine_range():
    a = torch.rand(2, 1, 32, 32)
    b = torch.rand(2, 1, 32, 32)
    val = ssim(a, b)
    assert -1.0 <= float(val) <= 1.0


def test_hybrid_loss_runs():
    crit = HybridDownscaleLoss(scale=2)
    hr = torch.rand(2, 3, 32, 32)
    outputs = {"lr": torch.rand(2, 3, 16, 16), "recon": torch.rand(2, 3, 32, 32)}
    out = crit(outputs, hr)
    assert "total" in out and out["total"].requires_grad is False or out["total"].dim() == 0
    assert "loss/mse" in out["logs"] and "loss/ssim" in out["logs"]


def test_hybrid_loss_weights():
    crit = HybridDownscaleLoss(lambda_mse=1.0, lambda_ssim=0.0, scale=2)
    hr = torch.rand(2, 3, 32, 32, requires_grad=False)
    recon = hr.clone().requires_grad_(True)
    outputs = {"lr": torch.rand(2, 3, 16, 16), "recon": recon}
    out = crit(outputs, hr)
    assert float(out["total"]) < 1e-5  # identical recon => ~0 MSE
