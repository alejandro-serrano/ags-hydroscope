"""Model factory: ResNet-50, EfficientNet-B0 and ViT-S/16 with 3 or 13 input bands.

Same function for the three models: only the architecture changes (docs/protocol.md section 4).
"""

import timm
import torch
import torch.nn as nn
from torchgeo.models import ResNet50_Weights, ViTSmall16_Weights, resnet50, vit_small_patch16_224

from hydroscope.data.eurosat import BAND_SETS

TIMM_NAMES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit_small": "vit_small_patch16_224",
}
BANDS = ("rgb", "ms13")
INITS = ("imagenet", "ssl4eo")

# SSL4EO-S12 weights were trained with the 13 L1C bands in this order (B8A right after B08).
# Source: torchgeo.models.resnet._sentinel2_toa_bands, used in the `meta["bands"]` of
# ResNet50_Weights.SENTINEL2_ALL_MOCO and ViTSmall16_Weights.SENTINEL2_ALL_DINO.
SSL4EO_BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)

# timm's pretrained first layer expects input channels (R, G, B). In EuroSAT order these bands
# are B04 (index 3), B03 (index 2) and B02 (index 1).
RGB_TO_MS13_INDEX = (3, 2, 1)


def get_first_layer(model: nn.Module, name: str) -> nn.Conv2d:
    """Return the first convolution (the patch embedding for the ViT) of a model."""
    if name == "resnet50":
        return model.conv1
    if name == "efficientnet_b0":
        return model.conv_stem
    return model.patch_embed.proj


def set_first_layer(model: nn.Module, name: str, layer: nn.Conv2d) -> None:
    """Replace the first convolution of a model by ``layer``."""
    if name == "resnet50":
        model.conv1 = layer
    elif name == "efficientnet_b0":
        model.conv_stem = layer
    else:
        model.patch_embed.proj = layer


def new_first_layer(old: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Return a conv like ``old`` but with ``in_channels`` input channels (random weights)."""
    return nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )


def rgb_to_ms13_weight(rgb_weight: torch.Tensor) -> torch.Tensor:
    """Expand first-layer weights (out, 3, k, k) to (out, 13, k, k), as in docs/protocol.md.

    The R, G, B weights go to B04, B03, B02 and every other band gets the mean of the three.
    No rescaling is applied (the protocol does not specify one).
    """
    mean_weight = rgb_weight.mean(dim=1, keepdim=True)  # (out, 1, k, k)
    channels = [mean_weight] * 13
    for rgb_index, ms13_index in enumerate(RGB_TO_MS13_INDEX):
        channels[ms13_index] = rgb_weight[:, rgb_index : rgb_index + 1]
    return torch.cat(channels, dim=1)  # (out, 13, k, k)


def ssl4eo_to_eurosat_weight(weight: torch.Tensor) -> torch.Tensor:
    """Reorder first-layer input channels from SSL4EO order to EuroSAT order (B8A last).

    (out, 13, k, k) -> (out, 13, k, k). Our tensors keep the EuroSAT band order for every model,
    so the permutation is done once here, on the weights, and not on the data.
    """
    eurosat_order = BAND_SETS["ms13"]
    source_index = [SSL4EO_BAND_ORDER.index(band) for band in eurosat_order]
    return weight[:, source_index]


def build_model(name: str, bands: str, init: str = "imagenet", num_classes: int = 10) -> nn.Module:
    """Build a classifier with a new ``num_classes`` head.

    Args:
        name: ``resnet50``, ``efficientnet_b0`` or ``vit_small``.
        bands: ``rgb`` (3 channels) or ``ms13`` (13 channels, EuroSAT band order).
        init: ``imagenet`` (timm weights) or ``ssl4eo`` (TorchGeo SSL4EO-S12, ms13 and
            resnet50 / vit_small only).
    """
    if name not in TIMM_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {list(TIMM_NAMES)}")
    if bands not in BANDS:
        raise ValueError(f"unknown bands {bands!r}; expected one of {list(BANDS)}")
    if init not in INITS:
        raise ValueError(f"unknown init {init!r}; expected one of {list(INITS)}")
    if init == "ssl4eo":
        return build_ssl4eo_model(name, bands, num_classes)

    # Load the 3-channel ImageNet model; timm's own in_chans adaptation is NOT used because it
    # rescales the weights and does not follow the protocol.
    model = timm.create_model(TIMM_NAMES[name], pretrained=True, num_classes=num_classes)
    if bands == "ms13":
        old = get_first_layer(model, name)
        layer = new_first_layer(old, 13)
        with torch.no_grad():
            layer.weight.copy_(rgb_to_ms13_weight(old.weight))
            if old.bias is not None:
                layer.bias.copy_(old.bias)
        set_first_layer(model, name, layer)
    return model


def build_ssl4eo_model(name: str, bands: str, num_classes: int) -> nn.Module:
    """Build resnet50 (MoCo) or vit_small (DINO) with SSL4EO-S12 weights and a new head."""
    if bands != "ms13":
        raise ValueError("init 'ssl4eo' needs bands 'ms13' (the weights use 13 bands)")
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO, num_classes=num_classes)
    elif name == "vit_small":
        model = vit_small_patch16_224(
            weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO, num_classes=num_classes
        )
    else:
        raise ValueError("init 'ssl4eo' is only available for resnet50 and vit_small")
    layer = get_first_layer(model, name)
    with torch.no_grad():
        layer.weight.copy_(ssl4eo_to_eurosat_weight(layer.weight))
    return model


def count_params(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    """Number of parameters with ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def last_block_params(model: nn.Module, name: str) -> list[nn.Parameter]:
    """Parameters trained in fine-tuning mode ``last_block`` (last block plus head).

    - resnet50: ``layer4`` (the last stage, 3 bottlenecks) and ``fc``.
    - efficientnet_b0: the last stage ``blocks[-1]``, ``conv_head``, ``bn2`` and ``classifier``.
    - vit_small: the last transformer block ``blocks[-1]``, the final ``norm`` and ``head``.
    """
    if name == "resnet50":
        modules = [model.layer4, model.fc]
    elif name == "efficientnet_b0":
        modules = [model.blocks[-1], model.conv_head, model.bn2, model.classifier]
    elif name == "vit_small":
        modules = [model.blocks[-1], model.norm, model.head]
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {list(TIMM_NAMES)}")
    params = []
    for module in modules:
        params += list(module.parameters())
    return params
