"""Grad-CAM maps (pytorch-grad-cam) and RGB overlays for the three models."""

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from hydroscope.data.transforms import normalize, resize_to_input, select_rgb


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Layer whose activations Grad-CAM uses: the last conv block (CNNs) or last norm (ViT)."""
    if name == "resnet50":
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.conv_head
    if name == "vit_small":
        return model.blocks[-1].norm1
    raise ValueError(f"unknown model {name!r}; expected resnet50, efficientnet_b0 or vit_small")


def vit_reshape_transform(tokens: torch.Tensor) -> torch.Tensor:
    """Turn ViT tokens into a feature map: drop the CLS token and fold 196 tokens into 14 x 14."""
    # (N, 197, 384) -> (N, 196, 384): token 0 is CLS
    patch_tokens = tokens[:, 1:, :]
    # (N, 196, 384) -> (N, 14, 14, 384)
    grid = patch_tokens.reshape(tokens.shape[0], 14, 14, tokens.shape[2])
    # (N, 14, 14, 384) -> (N, 384, 14, 14), channels first like a CNN feature map
    return grid.permute(0, 3, 1, 2)


def prepare_patch(
    raw: torch.Tensor, bands: str, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Raw 13-band patch (13, 64, 64) -> model input (C, 64, 64), normalized like in training."""
    if bands == "rgb":
        raw = select_rgb(raw)  # (13, 64, 64) -> (3, 64, 64)
    return normalize(raw, mean, std)


def min_max_scale(cam: np.ndarray) -> np.ndarray:
    """Scale a map to [0, 1]. A constant map becomes all zeros."""
    low = cam.min()
    high = cam.max()
    if high - low < 1e-12:
        return np.zeros_like(cam)
    return (cam - low) / (high - low)


def compute_gradcam(
    model: nn.Module, name: str, image: torch.Tensor, input_size: int = 224
) -> tuple[np.ndarray, int]:
    """Grad-CAM of the predicted class for one normalized patch (C, 64, 64).

    Returns the map as a (64, 64) array in [0, 1] and the predicted class index.
    """
    patch_size = image.shape[-1]
    batch = resize_to_input(image.unsqueeze(0), input_size)  # (1, C, 224, 224)
    model.eval()
    with torch.no_grad():
        pred = int(model(batch).argmax(dim=1).item())

    target_layer = get_target_layer(model, name)
    if name == "vit_small":
        cam_method = GradCAM(
            model=model, target_layers=[target_layer], reshape_transform=vit_reshape_transform
        )
    else:
        cam_method = GradCAM(model=model, target_layers=[target_layer])
    with cam_method as cam_engine:
        cam_batch = cam_engine(input_tensor=batch, targets=[ClassifierOutputTarget(pred)])
    cam = torch.from_numpy(cam_batch[0])  # (224, 224)

    # (224, 224) -> (1, 1, 224, 224) -> (1, 1, 64, 64) -> (64, 64)
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(patch_size, patch_size),
        mode="bilinear",
        align_corners=False,
    )
    return min_max_scale(cam[0, 0].numpy()), pred


def rgb_limits(patches: np.ndarray) -> np.ndarray:
    """2nd and 98th percentile of B04, B03, B02 over many raw patches (N, 13, 64, 64).

    Returns an array (3, 2): one (low, high) pair per RGB band.
    """
    rgb = select_rgb(torch.from_numpy(patches.astype(np.float32))).numpy()  # (N, 3, 64, 64)
    limits = np.zeros((3, 2), dtype=np.float32)
    for band in range(3):
        limits[band, 0] = np.percentile(rgb[:, band], 2)
        limits[band, 1] = np.percentile(rgb[:, band], 98)
    return limits


def to_rgb_image(raw: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Raw patch (13, 64, 64) -> RGB image (64, 64, 3) in [0, 1] with the shared stretch."""
    rgb = select_rgb(torch.from_numpy(raw.astype(np.float32))).numpy()  # (3, 64, 64)
    image = np.zeros((64, 64, 3), dtype=np.float32)
    for band in range(3):
        low, high = limits[band]
        image[:, :, band] = np.clip((rgb[band] - low) / (high - low), 0.0, 1.0)
    return image


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a (64, 64) map in [0, 1] (jet colormap) over an RGB image (64, 64, 3)."""
    colors = matplotlib.colormaps["jet"](cam)[:, :, :3]  # (64, 64, 3)
    return (1 - alpha) * rgb + alpha * colors
