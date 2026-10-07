"""Hydroscope Aguascalientes: CNNs vs. ViT for Sentinel-2 land-use classification."""

import os
import sys

__version__ = "0.1.0"

TORCH_IMPORTED_BEFORE_MPS_FALLBACK = "torch" in sys.modules
"""True if torch was imported before ``PYTORCH_ENABLE_MPS_FALLBACK`` could be set here."""

# Ops without an MPS kernel then run on the CPU instead of raising. The variable is read when
# torch initializes, so it must be set before the first ``import torch``. A value already set by
# the user wins. ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` is deliberately left untouched.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
