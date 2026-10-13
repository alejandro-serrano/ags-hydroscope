"""Model factory: shapes, first-layer adaptation, SSL4EO band order and last-block parameters."""

import pytest
import timm
import torch
from torchgeo.models import ResNet50_Weights, ViTSmall16_Weights, resnet50, vit_small_patch16_224

from hydroscope.models.factory import (
    TIMM_NAMES,
    build_model,
    count_params,
    count_trainable_params,
    get_first_layer,
    last_block_params,
)

VALID = [
    ("resnet50", "rgb", "imagenet"),
    ("efficientnet_b0", "rgb", "imagenet"),
    ("vit_small", "rgb", "imagenet"),
    ("resnet50", "ms13", "imagenet"),
    ("efficientnet_b0", "ms13", "imagenet"),
    ("vit_small", "ms13", "imagenet"),
    ("resnet50", "ms13", "ssl4eo"),
    ("vit_small", "ms13", "ssl4eo"),
]
MODELS = {}


def get_model(name: str, bands: str, init: str) -> torch.nn.Module:
    """Build each model once per test module."""
    key = (name, bands, init)
    if key not in MODELS:
        MODELS[key] = build_model(name, bands, init)
    return MODELS[key]


def n_bands(bands: str) -> int:
    return 3 if bands == "rgb" else 13


@pytest.mark.parametrize("name,bands,init", VALID)
def test_build_model_returns_10_logits(name: str, bands: str, init: str) -> None:
    model = get_model(name, bands, init).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, n_bands(bands), 224, 224))
    assert out.shape == (2, 10)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_build_model_forward_on_mps() -> None:
    model = get_model("resnet50", "ms13", "imagenet").to("mps").eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 13, 224, 224, device="mps"))
    assert out.shape == (2, 10)
    model.to("cpu")


@pytest.mark.parametrize("name", list(TIMM_NAMES))
def test_ms13_first_layer_copies_rgb_weights_to_b04_b03_b02(name: str) -> None:
    rgb = timm.create_model(TIMM_NAMES[name], pretrained=True)
    w_rgb = get_first_layer(rgb, name).weight  # (out, 3, k, k)
    w_ms = get_first_layer(get_model(name, "ms13", "imagenet"), name).weight  # (out, 13, k, k)
    assert torch.equal(w_ms[:, 3], w_rgb[:, 0])  # B04 <- R
    assert torch.equal(w_ms[:, 2], w_rgb[:, 1])  # B03 <- G
    assert torch.equal(w_ms[:, 1], w_rgb[:, 2])  # B02 <- B


def test_ms13_other_bands_get_the_mean_of_rgb_weights() -> None:
    rgb = timm.create_model("resnet50", pretrained=True)
    mean = rgb.conv1.weight.mean(dim=1)
    w_ms = get_model("resnet50", "ms13", "imagenet").conv1.weight
    for index in [0, 4, 7, 8, 12]:
        assert torch.allclose(w_ms[:, index], mean, atol=1e-6)


def test_ms13_keeps_the_pretrained_rgb_model_untouched_elsewhere() -> None:
    rgb = timm.create_model("resnet50", pretrained=True)
    ms = get_model("resnet50", "ms13", "imagenet")
    assert torch.equal(rgb.layer4[0].conv1.weight, ms.layer4[0].conv1.weight)


def test_ssl4eo_resnet_first_layer_is_permuted_to_eurosat_order() -> None:
    raw = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO, num_classes=10)
    w_raw = raw.conv1.weight  # SSL4EO order: B8A is channel 8
    w_ours = get_model(
        "resnet50", "ms13", "ssl4eo"
    ).conv1.weight  # EuroSAT order: B8A is channel 12
    assert torch.equal(w_ours[:, 12], w_raw[:, 8])  # B8A
    assert torch.equal(w_ours[:, 8], w_raw[:, 9])  # B09
    assert torch.equal(w_ours[:, 11], w_raw[:, 12])  # B12
    assert torch.equal(w_ours[:, 7], w_raw[:, 7])  # B08 unchanged


def test_ssl4eo_vit_patch_embed_is_permuted_to_eurosat_order() -> None:
    raw = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO, num_classes=10)
    w_raw = raw.patch_embed.proj.weight
    w_ours = get_model("vit_small", "ms13", "ssl4eo").patch_embed.proj.weight
    assert torch.equal(w_ours[:, 12], w_raw[:, 8])
    assert torch.equal(w_ours[:, 9], w_raw[:, 10])


def test_build_model_is_deterministic_for_the_new_head_with_fixed_seed() -> None:
    torch.manual_seed(0)
    a = build_model("resnet50", "rgb", "imagenet")
    torch.manual_seed(0)
    b = build_model("resnet50", "rgb", "imagenet")
    assert torch.equal(a.fc.weight, b.fc.weight)


def test_invalid_values_raise_value_error() -> None:
    with pytest.raises(ValueError):
        build_model("vgg16", "rgb", "imagenet")
    with pytest.raises(ValueError):
        build_model("resnet50", "nir", "imagenet")
    with pytest.raises(ValueError):
        build_model("resnet50", "rgb", "random")


def test_ssl4eo_invalid_combinations_raise_value_error() -> None:
    with pytest.raises(ValueError):
        build_model("resnet50", "rgb", "ssl4eo")
    with pytest.raises(ValueError):
        build_model("efficientnet_b0", "ms13", "ssl4eo")


@pytest.mark.parametrize("name", list(TIMM_NAMES))
def test_last_block_params_is_a_non_empty_strict_subset(name: str) -> None:
    model = get_model(name, "rgb", "imagenet")
    selected = sum(p.numel() for p in last_block_params(model, name))
    assert 0 < selected < count_params(model)


def test_last_block_params_resnet50_are_layer4_and_fc() -> None:
    model = get_model("resnet50", "rgb", "imagenet")
    selected = {id(p) for p in last_block_params(model, "resnet50")}
    assert id(model.fc.weight) in selected and id(model.layer4[0].conv1.weight) in selected
    assert id(model.layer3[0].conv1.weight) not in selected


def test_count_params_matches_trainable_when_nothing_is_frozen() -> None:
    model = get_model("efficientnet_b0", "rgb", "imagenet")
    assert count_params(model) == count_trainable_params(model)
