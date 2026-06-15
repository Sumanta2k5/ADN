import torch

from adn.models import build_adn
from adn.models.attention import CBAM, ARM
from adn.utils.config import Config


def _cfg(scale=2, **model_kwargs):
    model = {"channels": 16, "num_feature_blocks": 2, "kernel_depth": 2,
             "offset_depth": 2, "feature_downscale": 8}
    model.update(model_kwargs)
    return Config({"scale": scale, "model": model})


def test_cbam_shape():
    x = torch.rand(2, 16, 8, 8)
    out = CBAM(16, reduction=4)(x)
    assert out.shape == x.shape


def test_arm_disabled_is_identity():
    x = torch.rand(2, 16, 8, 8)
    arm = ARM(16, enabled=False)
    assert torch.equal(arm(x), x)


def test_adn_forward_x2():
    model = build_adn(_cfg(scale=2))
    hr = torch.rand(1, 3, 64, 64)
    out = model(hr)
    assert out["lr"].shape == (1, 3, 32, 32)
    assert out["recon"].shape == (1, 3, 64, 64)


def test_adn_forward_x4_and_params():
    model = build_adn(_cfg(scale=4))
    hr = torch.rand(1, 3, 64, 64)
    out = model(hr, return_params=True)
    assert out["lr"].shape == (1, 3, 16, 16)
    assert "kernels" in out and "offset_x" in out


def test_adn_backward():
    model = build_adn(_cfg(scale=2))
    hr = torch.rand(1, 3, 64, 64)
    out = model(hr)
    loss = out["recon"].mean()
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_wo_resnet_builds():
    model = build_adn(_cfg(scale=2, use_resnet=False))
    out = model(torch.rand(1, 3, 64, 64))
    assert out["lr"].shape == (1, 3, 32, 32)
