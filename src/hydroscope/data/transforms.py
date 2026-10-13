"""Normalization, augmentation and resizing of EuroSAT / Aguascalientes patches.

All functions work on raw digital numbers (float32). The data pipeline keeps the EuroSAT band
order for every model.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hydroscope.data.eurosat import BAND_SETS

RGB_INDEX = [3, 2, 1]
"""Positions of B04, B03, B02 in a 13-band EuroSAT-order tensor."""

# TorchGeo normalization of the SSL4EO-S12 weights (ResNet50_Weights.SENTINEL2_ALL_MOCO and
# ViTSmall16_Weights.SENTINEL2_ALL_DINO): T.Normalize(mean=[0], std=[10000]) on raw reflectance,
# i.e. x / 10000 for all 13 bands (torchgeo.models.resnet._ssl4eo_s12_transforms_s2_10k and
# torchgeo.models.vit._zhu_xlab_transforms).
SSL4EO_MEAN = 0.0
SSL4EO_STD = 10000.0


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-band normalization. x is (C, H, W) or (N, C, H, W); mean and std are (C,)."""
    mean = mean.view(-1, 1, 1)  # (C, 1, 1) broadcasts over both layouts
    std = std.view(-1, 1, 1)
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse of ``normalize``."""
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return x * std + mean


def load_band_stats(path: str | Path, bands: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read per-band mean and std of the EuroSAT train split; returns two (C,) tensors.

    The JSON maps band name to ``{"mean": ..., "std": ...}``. Bands are selected and ordered
    following ``BAND_SETS[bands]`` (rgb gives B04, B03, B02).
    """
    stats = json.loads(Path(path).read_text())
    band_names = BAND_SETS[bands]
    mean = [stats[band]["mean"] for band in band_names]
    std = [stats[band]["std"] for band in band_names]
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def get_mean_std(
    init: str, bands: str, stats_path: str | Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (C,) mean and std that go with a weight initialization.

    imagenet -> EuroSAT train statistics; ssl4eo -> TorchGeo's x / 10000 (13 bands only).
    """
    if init == "imagenet":
        return load_band_stats(stats_path, bands)
    if init == "ssl4eo":
        if bands != "ms13":
            raise ValueError("init 'ssl4eo' needs bands 'ms13'")
        return torch.full((13,), SSL4EO_MEAN), torch.full((13,), SSL4EO_STD)
    raise ValueError(f"unknown init {init!r}; expected 'imagenet' or 'ssl4eo'")


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random horizontal flip, vertical flip and 90-degree rotation (k in 0..3).

    Meant for ONE sample (C, H, W), called inside the training Dataset; the random choices come
    from ``generator`` so a fixed seed gives the same result. It also accepts (N, C, H, W), but
    then the same transform is applied to the whole batch. Only the training split is augmented:
    the caller must not use it for val or test.
    """
    if torch.rand(1, generator=generator).item() < 0.5:
        x = torch.flip(x, dims=[-1])  # horizontal flip: mirror columns
    if torch.rand(1, generator=generator).item() < 0.5:
        x = torch.flip(x, dims=[-2])  # vertical flip: mirror rows
    k = int(torch.randint(0, 4, (1,), generator=generator).item())
    return torch.rot90(x, k, dims=(-2, -1))  # square patches keep their shape


def resize_to_input(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Bilinear resize of a batch: (N, C, 64, 64) -> (N, C, size, size). Do it on the device."""
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def select_rgb(x: torch.Tensor) -> torch.Tensor:
    """Pick B04, B03, B02: (13, H, W) -> (3, H, W) or (N, 13, H, W) -> (N, 3, H, W)."""
    return x[..., RGB_INDEX, :, :]
