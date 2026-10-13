"""Building blocks shared by every training run: accumulation, resizing, loading, CLI."""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from hydroscope.data.eurosat import EuroSATDataset
from hydroscope.data.transforms import augment, normalize


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
    worker_init_fn=None,
) -> DataLoader:
    """DataLoader with a seeded shuffling generator.

    Persistent workers are used only when ``num_workers > 0``; ``pin_memory`` is off (unified
    memory on Apple silicon). ``worker_init_fn`` runs once in every worker (e.g. ``seed_worker``).
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
        worker_init_fn=worker_init_fn,
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


class NormalizedDataset(Dataset):
    """Wraps a dataset of ``{"image", "label"}`` dicts and returns ``(image, label)`` tuples.

    The image is normalized per band; if ``train`` is true it is also flipped / rotated with a
    seeded generator. With ``num_workers > 0`` every worker gets a copy of the dataset, so the
    loader must use ``seed_worker`` to give each copy of the generator its own seed.
    """

    def __init__(self, base, mean, std, train: bool, seed: int = 0):
        self.base = base
        self.mean = mean  # (C,)
        self.std = std  # (C,)
        self.train = train
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        image = normalize(sample["image"], self.mean, self.std)  # (C, 64, 64)
        if self.train:
            image = augment(image, self.generator)
        return image, sample["label"]


def seed_worker(worker_id: int) -> None:
    """Re-seed the augmentation generator of one DataLoader worker (used as ``worker_init_fn``).

    Without this, every worker would draw the same flips and rotations. PyTorch seeds each worker
    with base_seed + worker_id, and base_seed comes from the loader's seeded generator, so
    ``torch.initial_seed()`` is different in every worker and the same in every run.
    """
    dataset = torch.utils.data.get_worker_info().dataset  # this worker's copy of the dataset
    dataset.generator.manual_seed(torch.initial_seed())


def make_eurosat_loader(cfg: dict, split: str, mean, std, train: bool) -> DataLoader:
    """DataLoader of one EuroSAT split built from a resolved run config.

    ``train=True`` shuffles and augments. ``data.max_samples`` keeps a seeded random subset of
    every split (used by the smoke test).
    """
    data_cfg = cfg["data"]
    base = EuroSATDataset(
        split, bands=cfg["bands"], root=data_cfg["root"], cache_dir=data_cfg["cache_dir"]
    )
    max_samples = data_cfg["max_samples"]
    if max_samples is not None and max_samples < len(base):
        generator = torch.Generator()
        generator.manual_seed(cfg["seed"])
        indices = torch.randperm(len(base), generator=generator)[:max_samples].tolist()
        base = Subset(base, indices)
    dataset = NormalizedDataset(base, mean, std, train=train, seed=cfg["seed"])
    return make_dataloader(
        dataset,
        cfg["optim"]["micro_batch_size"],
        shuffle=train,
        num_workers=data_cfg["num_workers"],
        seed=cfg["seed"],
        worker_init_fn=seed_worker,
    )
