"""Normalization, augmentation, resize and RGB selection."""

from pathlib import Path

import torch

from hydroscope.data.transforms import (
    augment,
    denormalize,
    get_mean_std,
    normalize,
    resize_to_input,
    select_rgb,
)

STATS = Path(__file__).resolve().parents[1] / "configs" / "stats" / "eurosat_train_stats.json"


def test_augment_same_seed_gives_same_result() -> None:
    x = torch.arange(3 * 4 * 4, dtype=torch.float32).view(3, 4, 4)
    for seed in range(5):
        a = augment(x, torch.Generator().manual_seed(seed))
        b = augment(x, torch.Generator().manual_seed(seed))
        assert torch.equal(a, b)


def test_augment_keeps_shape_and_values() -> None:
    x = torch.arange(3 * 4 * 4, dtype=torch.float32).view(3, 4, 4)
    out = augment(x, torch.Generator().manual_seed(1))
    assert out.shape == x.shape
    assert torch.equal(out.flatten().sort().values, x.flatten().sort().values)


def test_augment_produces_different_outputs_across_seeds() -> None:
    x = torch.arange(1 * 4 * 4, dtype=torch.float32).view(1, 4, 4)
    outputs = [augment(x, torch.Generator().manual_seed(s)) for s in range(20)]
    assert any(not torch.equal(o, x) for o in outputs)


def test_normalize_then_denormalize_recovers_input() -> None:
    # Values near 1: float32 rounding of raw reflectances (~5000) alone exceeds 1e-5.
    x = torch.rand(2, 13, 8, 8)
    mean = torch.rand(13)
    std = torch.rand(13) + 0.5
    back = denormalize(normalize(x, mean, std), mean, std)
    assert torch.allclose(back, x, atol=1e-5, rtol=1e-5)


def test_normalize_works_on_a_single_sample() -> None:
    x = torch.ones(3, 8, 8)
    mean = torch.tensor([1.0, 0.0, 1.0])
    std = torch.tensor([2.0, 1.0, 1.0])
    out = normalize(x, mean, std)
    assert out.shape == (3, 8, 8)
    assert out[0, 0, 0] == 0.0 and out[1, 0, 0] == 1.0


def test_mean_std_imagenet_rgb_uses_b04_b03_b02() -> None:
    mean_rgb, _ = get_mean_std("imagenet", "rgb", STATS)
    mean_ms13, _ = get_mean_std("imagenet", "ms13", STATS)
    assert mean_rgb.tolist() == [mean_ms13[3].item(), mean_ms13[2].item(), mean_ms13[1].item()]


def test_mean_std_ssl4eo_is_divide_by_10000() -> None:
    mean, std = get_mean_std("ssl4eo", "ms13", STATS)
    assert mean.shape == (13,) and bool((mean == 0).all()) and bool((std == 10000).all())


def test_select_rgb_picks_indices_3_2_1() -> None:
    x = torch.arange(13, dtype=torch.float32).view(13, 1, 1).expand(13, 2, 2)
    assert select_rgb(x)[:, 0, 0].tolist() == [3.0, 2.0, 1.0]
    batch = x.unsqueeze(0).repeat(2, 1, 1, 1)
    assert select_rgb(batch).shape == (2, 3, 2, 2)
    assert select_rgb(batch)[1, :, 0, 0].tolist() == [3.0, 2.0, 1.0]


def test_resize_to_input_gives_224() -> None:
    x = torch.zeros(2, 13, 64, 64)
    assert resize_to_input(x).shape == (2, 13, 224, 224)
