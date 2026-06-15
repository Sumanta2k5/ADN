import torch

from adn.models.resampler import adaptive_resample
from adn.utils.imresize import imresize


def test_resampler_output_shape():
    b, c, h, w = 2, 3, 32, 32
    scale, k = 2, 3
    hl, wl = h // scale, w // scale
    hr = torch.rand(b, c, h, w)
    kernels = torch.softmax(torch.rand(b, k * k, hl, wl), dim=1)
    dx = torch.zeros(b, k * k, hl, wl)
    dy = torch.zeros(b, k * k, hl, wl)
    out = adaptive_resample(hr, kernels, dx, dy, scale, k)
    assert out.shape == (b, c, hl, wl)


def test_resampler_preserves_brightness_constant_image():
    # A constant image resampled by a normalized kernel stays constant.
    b, c, h, w = 1, 3, 16, 16
    scale, k = 2, 3
    hl, wl = h // scale, w // scale
    hr = torch.full((b, c, h, w), 0.5)
    kernels = torch.softmax(torch.zeros(b, k * k, hl, wl), dim=1)  # uniform
    dx = torch.zeros(b, k * k, hl, wl)
    dy = torch.zeros(b, k * k, hl, wl)
    out = adaptive_resample(hr, kernels, dx, dy, scale, k)
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5)


def test_resampler_is_differentiable():
    b, c, h, w = 1, 3, 16, 16
    scale, k = 2, 3
    hl, wl = h // scale, w // scale
    hr = torch.rand(b, c, h, w)
    logits = torch.rand(b, k * k, hl, wl, requires_grad=True)
    kernels = torch.softmax(logits, dim=1)
    dx = torch.zeros(b, k * k, hl, wl, requires_grad=True)
    dy = torch.zeros(b, k * k, hl, wl, requires_grad=True)
    out = adaptive_resample(hr, kernels, dx, dy, scale, k)
    out.mean().backward()
    assert logits.grad is not None and dx.grad is not None


def test_imresize_matches_size():
    img = torch.rand(3, 40, 40)
    down = imresize(img, scale=0.5)
    assert down.shape[-2:] == (20, 20)
    up = imresize(down, sizes=(40, 40))
    assert up.shape[-2:] == (40, 40)
