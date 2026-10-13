"""Grad-CAM: map size and range for the three models, and the ViT token reshape."""

import numpy as np
import torch

from hydroscope.explain.gradcam import compute_gradcam, min_max_scale, vit_reshape_transform
from hydroscope.models.factory import build_model

MODELS = {}


def get_model(name: str) -> torch.nn.Module:
    """Build each rgb/imagenet model once per test module."""
    if name not in MODELS:
        MODELS[name] = build_model(name, "rgb", "imagenet").eval()
    return MODELS[name]


def check_cam(name: str) -> None:
    torch.manual_seed(0)
    image = torch.randn(3, 64, 64)
    cam, pred = compute_gradcam(get_model(name), name, image)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert 0 <= pred < 10


def test_resnet50_gradcam_has_patch_size_and_values_in_0_1() -> None:
    check_cam("resnet50")


def test_efficientnet_b0_gradcam_has_patch_size_and_values_in_0_1() -> None:
    check_cam("efficientnet_b0")


def test_vit_small_gradcam_has_patch_size_and_values_in_0_1() -> None:
    check_cam("vit_small")


def test_gradcam_is_deterministic() -> None:
    torch.manual_seed(0)
    image = torch.randn(3, 64, 64)
    cam_a, _ = compute_gradcam(get_model("resnet50"), "resnet50", image)
    cam_b, _ = compute_gradcam(get_model("resnet50"), "resnet50", image)
    assert np.array_equal(cam_a, cam_b)


def test_vit_reshape_transform_drops_cls_and_makes_14x14_map() -> None:
    out = vit_reshape_transform(torch.zeros(2, 197, 384))
    assert out.shape == (2, 384, 14, 14)


def test_vit_reshape_transform_keeps_token_order() -> None:
    tokens = torch.arange(197).float().view(1, 197, 1).repeat(1, 1, 384)
    out = vit_reshape_transform(tokens)
    assert out[0, 0, 0, 0] == 1.0  # first patch token, CLS (0) dropped
    assert out[0, 0, 0, 1] == 2.0  # next token is the next column


def test_min_max_scale_constant_map_is_zeros() -> None:
    assert np.array_equal(min_max_scale(np.full((4, 4), 3.0)), np.zeros((4, 4)))
