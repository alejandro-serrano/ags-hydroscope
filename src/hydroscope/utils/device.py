"""Device selection, MPS CPU-fallback logging and peak-memory tracking.

All runs use fp32 and no autocast on every device (see ``PRECISION``), so that the three models
are trained under identical numerics.
"""

from __future__ import annotations

import logging
import os
import warnings

import torch

import hydroscope

PRECISION = "fp32"
"""Numeric precision of every run. No ``torch.autocast`` anywhere."""

MPS_FALLBACK_PATTERN = r".*fall back to run on the CPU.*"
"""Regex matching the warning PyTorch emits when an MPS op falls back to the CPU."""

_BYTES_PER_MB = 1024**2


def select_device(preferred: str = "auto") -> torch.device:
    """Pick the compute device.

    Args:
        preferred: ``"auto"`` (cuda, then mps, then cpu) or an explicit ``"cuda"``, ``"mps"``
            or ``"cpu"``.

    Raises:
        ValueError: if ``preferred`` is not one of the accepted names.
        RuntimeError: if an explicitly requested accelerator is not available.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"unknown device '{preferred}'; expected auto, cuda, mps or cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if preferred == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(preferred)


def log_mps_fallbacks(logger: logging.Logger | None = None) -> None:
    """Route PyTorch's "fall back to run on the CPU" warnings to ``logging``.

    Every MPS op that falls back to the CPU is then logged as a warning on the ``py.warnings``
    logger instead of being shown once and forgotten. Also warns on ``logger`` if torch was
    imported before ``PYTORCH_ENABLE_MPS_FALLBACK`` was set, in which case the fallback may be
    inactive.
    """
    logger = logger or logging.getLogger(__name__)
    logging.captureWarnings(True)
    warnings.filterwarnings("always", message=MPS_FALLBACK_PATTERN)
    if (
        hydroscope.TORCH_IMPORTED_BEFORE_MPS_FALLBACK
        or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
    ):
        logger.warning(
            "PYTORCH_ENABLE_MPS_FALLBACK was not set to 1 before torch was imported; "
            "unsupported MPS ops may raise instead of falling back to the CPU."
        )


class PeakMemoryTracker:
    """Peak device memory of a run, in MB.

    Call :meth:`update` once per training step. On MPS the peak is the maximum of
    ``torch.mps.driver_allocated_memory()`` sampled at each update; on CUDA it is
    ``torch.cuda.max_memory_allocated()``; on CPU it is ``None``.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._peak_bytes = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def update(self) -> None:
        """Sample the current device memory."""
        if self.device.type == "mps":
            self._peak_bytes = max(self._peak_bytes, torch.mps.driver_allocated_memory())
        elif self.device.type == "cuda":
            self._peak_bytes = max(self._peak_bytes, torch.cuda.max_memory_allocated(self.device))

    @property
    def peak_mb(self) -> float | None:
        """Peak memory in MB (1 MB = 2**20 bytes), or ``None`` on CPU."""
        if self.device.type not in {"mps", "cuda"}:
            return None
        self.update()
        return self._peak_bytes / _BYTES_PER_MB
