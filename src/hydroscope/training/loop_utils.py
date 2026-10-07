"""Building blocks shared by every training run: accumulation, resizing, loading, CLI."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def accumulation_steps(batch_size: int, micro_batch_size: int) -> int:
    """Number of micro-batches per optimizer step (effective batch = steps * micro batch).

    Raises:
        ValueError: if either size is not positive or ``micro_batch_size`` does not divide
            ``batch_size``.
    """
    if batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive")
    if batch_size % micro_batch_size != 0:
        raise ValueError(
            f"micro_batch_size {micro_batch_size} does not divide batch_size {batch_size}"
        )
    return batch_size // micro_batch_size


def resize_batch(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Bilinearly resize a ``(N, C, H, W)`` batch to ``(N, C, size, size)`` on its own device."""
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def make_dataloader(
    dataset: Dataset,
    micro_batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    seed: int = 0,
) -> DataLoader:
    """DataLoader with a seeded shuffling generator.

    Persistent workers are used only when ``num_workers > 0``; ``pin_memory`` is off (unified
    memory on Apple silicon).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        generator=generator,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI shared by the training entry points.

    ``--max-steps`` overrides ``schedule.max_steps`` of the config (benchmarking only).
    """
    parser = argparse.ArgumentParser(description="Hydroscope training run")
    parser.add_argument("--config", required=True, help="path to the run YAML config")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many optimizer steps (overrides schedule.max_steps)",
    )
    return parser
