"""EuroSAT (TorchGeo MS, official split): dataset wrapper, .npy cache and band statistics.

Samples are returned as raw digital numbers (float32); normalization and augmentation are applied
later in the pipeline. The cache stores each split as one ``uint16`` array so that training reads
from a memory-mapped file instead of 27,000 GeoTIFFs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchgeo.datasets import EuroSAT

BAND_SETS: dict[str, tuple[str, ...]] = {
    "ms13": tuple(EuroSAT.all_band_names),
    "rgb": ("B04", "B03", "B02"),
}
"""Input band sets: all 13 bands in TorchGeo order (B8A last) and RGB (B04, B03, B02)."""

assert BAND_SETS["ms13"][-1] == "B8A" and len(BAND_SETS["ms13"]) == 13

CLASSES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
"""The 10 classes in TorchGeo index order (docs/protocol.md section 3)."""

SPLITS = ("train", "val", "test")
PATCH_SHAPE = (13, 64, 64)
"""Shape of one cached ms13 patch (bands, rows, cols)."""


def cache_paths(cache_dir: str | Path, split: str) -> tuple[Path, Path]:
    """Return the ``(x, y)`` cache file paths of a split."""
    cache_dir = Path(cache_dir)
    return cache_dir / f"{split}_x.npy", cache_dir / f"{split}_y.npy"


class EuroSATDataset(Dataset):
    """EuroSAT split returning ``{"image": float32 (C, 64, 64), "label": int64 scalar}``.

    If ``{cache_dir}/{split}_x.npy`` and ``{split}_y.npy`` exist they are read through
    ``np.load(mmap_mode="r")`` and the requested bands are selected by index. Otherwise samples
    come from TorchGeo (downloading the data if ``download`` is true). Values are raw digital
    numbers in both cases.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.
        bands: key of ``BAND_SETS`` (``"ms13"`` or ``"rgb"``).
        root: TorchGeo dataset root.
        cache_dir: directory of the ``.npy`` cache; defaults to ``{root}/cache``.
        download: let TorchGeo download the dataset when the cache is absent.
        base_cls: TorchGeo dataset class; tests pass ``EuroSAT100``.
    """

    def __init__(
        self,
        split: str,
        bands: str = "ms13",
        root: str | Path = "data/eurosat",
        cache_dir: str | Path | None = None,
        download: bool = True,
        base_cls: type[EuroSAT] = EuroSAT,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split '{split}'; expected one of {SPLITS}")
        if bands not in BAND_SETS:
            raise ValueError(f"unknown bands '{bands}'; expected one of {tuple(BAND_SETS)}")
        self.split = split
        self.bands = bands
        self.band_names = BAND_SETS[bands]
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.root / "cache"
        self._band_idx = np.array(
            [BAND_SETS["ms13"].index(b) for b in self.band_names], dtype=np.int64
        )
        self._x_path, self._y_path = cache_paths(self.cache_dir, split)
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._base: EuroSAT | None = None
        if self._x_path.exists() and self._y_path.exists():
            y = np.load(self._y_path, mmap_mode="r")
            self._len = len(y)
            self.classes = list(CLASSES)
        else:
            self._base = base_cls(self.root, split, bands=self.band_names, download=download)
            self._len = len(self._base)
            self.classes = list(self._base.classes)

    @property
    def from_cache(self) -> bool:
        """True if samples are read from the ``.npy`` cache."""
        return self._base is None

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._base is not None:
            sample = self._base[index]
            return {
                "image": sample["image"].float(),
                "label": torch.as_tensor(sample["label"], dtype=torch.long),
            }
        if self._x is None or self._y is None:
            # Opened lazily so that DataLoader workers map the files themselves.
            self._x = np.load(self._x_path, mmap_mode="r")
            self._y = np.load(self._y_path, mmap_mode="r")
        image = np.asarray(self._x[index])[self._band_idx].astype(np.float32)
        return {
            "image": torch.from_numpy(image),
            "label": torch.tensor(int(self._y[index]), dtype=torch.long),
        }

    def __getstate__(self) -> dict[str, Any]:
        # Never pickle the memory maps (that would copy the whole array into each worker).
        state = self.__dict__.copy()
        state["_x"] = None
        state["_y"] = None
        return state


def write_split_cache(
    split: str,
    root: str | Path,
    cache_dir: str | Path,
    base_cls: type[EuroSAT] = EuroSAT,
    download: bool = True,
) -> tuple[Path, Path]:
    """Stream a TorchGeo split (13 bands) into ``{split}_x.npy`` / ``{split}_y.npy``.

    ``x`` is ``uint16`` of shape ``(N, 13, 64, 64)`` and ``y`` is ``uint16`` of shape ``(N,)``.
    Files are written through ``np.lib.format.open_memmap``
    to temporary names and renamed at the end, so an interrupted run leaves no partial cache.

    Raises:
        ValueError: if a sample is not a ``(13, 64, 64)`` array of integers in [0, 65535], i.e.
            if the ``uint16`` cast would not be lossless.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = cache_paths(cache_dir, split)
    base = base_cls(Path(root), split, bands=BAND_SETS["ms13"], download=download)
    n = len(base)
    x_tmp = x_path.with_name(x_path.name + ".tmp")
    y_tmp = y_path.with_name(y_path.name + ".tmp")
    x = np.lib.format.open_memmap(x_tmp, mode="w+", dtype=np.uint16, shape=(n, *PATCH_SHAPE))
    y = np.lib.format.open_memmap(y_tmp, mode="w+", dtype=np.uint16, shape=(n,))
    for i in range(n):
        sample = base[i]
        image = sample["image"].numpy()
        if image.shape != PATCH_SHAPE:
            raise ValueError(f"{split}[{i}]: expected shape {PATCH_SHAPE}, got {image.shape}")
        if not (np.isfinite(image).all() and (image == np.rint(image)).all()):
            raise ValueError(f"{split}[{i}]: non-integer values, uint16 cast would be lossy")
        if image.min() < 0 or image.max() > np.iinfo(np.uint16).max:
            raise ValueError(f"{split}[{i}]: values outside [0, 65535]")
        x[i] = image.astype(np.uint16)
        y[i] = int(sample["label"])
    x.flush()
    y.flush()
    del x, y
    os.replace(x_tmp, x_path)
    os.replace(y_tmp, y_path)
    return x_path, y_path


def compute_band_stats(
    batches: Iterable[Any], band_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per-band mean and population std over all pixels, streaming in float64.

    Each batch is an array-like of shape ``(N, C, H, W)`` with ``C == len(band_names)``. Batch
    moments are merged with the parallel (Chan et al.) update, so the result does not depend on
    how the data is chunked.

    Returns:
        ``{band: {"mean": float, "std": float}}`` keyed by band name.

    Raises:
        ValueError: if there are no batches or a batch has the wrong shape.
    """
    n_bands = len(band_names)
    count = 0
    mean = np.zeros(n_bands, dtype=np.float64)
    m2 = np.zeros(n_bands, dtype=np.float64)
    for batch in batches:
        arr = np.asarray(batch, dtype=np.float64)
        if arr.ndim != 4 or arr.shape[1] != n_bands:
            raise ValueError(f"expected batches of shape (N, {n_bands}, H, W), got {arr.shape}")
        n_b = arr.shape[0] * arr.shape[2] * arr.shape[3]
        if n_b == 0:
            continue
        mean_b = arr.mean(axis=(0, 2, 3))
        m2_b = ((arr - mean_b[None, :, None, None]) ** 2).sum(axis=(0, 2, 3))
        total = count + n_b
        delta = mean_b - mean
        mean = mean + delta * (n_b / total)
        m2 = m2 + m2_b + delta**2 * (count * n_b / total)
        count = total
    if count == 0:
        raise ValueError("no pixels to compute statistics from")
    std = np.sqrt(m2 / count)
    return {
        band: {"mean": float(mean[i]), "std": float(std[i])} for i, band in enumerate(band_names)
    }


def _iter_chunks(x: np.ndarray, chunk: int) -> Iterator[np.ndarray]:
    for start in range(0, len(x), chunk):
        yield x[start : start + chunk]


def train_band_stats(cache_dir: str | Path, chunk: int = 512) -> dict[str, dict[str, float]]:
    """Per-band statistics of the ms13 train split, read only from ``train_x.npy``.

    The val and test caches are never opened, so no statistic leaks from them.
    """
    x_path, _ = cache_paths(cache_dir, "train")
    x = np.load(x_path, mmap_mode="r")
    return compute_band_stats(_iter_chunks(x, chunk), BAND_SETS["ms13"])


def save_band_stats(stats: Mapping[str, Mapping[str, float]], path: str | Path) -> None:
    """Write band statistics as JSON keyed by band name (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n")
