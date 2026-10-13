"""Aguascalientes dataset: labeled 13-band patches (contracts 1 and 2 in docs/contracts.md)."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from hydroscope.data.eurosat import CLASSES
from hydroscope.data.labels import SKIP, load_labels
from hydroscope.data.transforms import augment, normalize, select_rgb

LABELS_CSV = Path("data/ags/labels.csv")
PATCHES_DIR = Path("data/ags/patches")
AGS_SPLITS = ("finetune", "test")


class AguascalientesDataset(Dataset):
    """Patches of one split (``finetune`` or ``test``) with their class index.

    All patches are read once into memory (about 600 patches of 13 x 64 x 64). ``skip`` rows are
    ignored. ``__getitem__`` returns the normalized ``(image, label)``; if ``train`` is true the
    image is also flipped / rotated with a seeded generator (as ``NormalizedDataset`` does).
    """

    def __init__(
        self,
        split: str,
        bands: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        train: bool,
        labels_csv: Path = LABELS_CSV,
        patches_dir: Path = PATCHES_DIR,
        seed: int = 0,
    ):
        if split not in AGS_SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {list(AGS_SPLITS)}")
        if bands not in ("rgb", "ms13"):
            raise ValueError("bands must be 'rgb' or 'ms13'")
        table = load_labels(labels_csv)
        table = table[(table["split"] == split) & (table["label"] != SKIP)]
        if len(table) == 0:
            raise ValueError(f"no labeled patches for split {split!r} in {labels_csv}")
        if table["cell_id"].duplicated().any():
            raise ValueError(f"duplicate cell_id in split {split!r} of {labels_csv}")

        self.cell_ids = table["cell_id"].tolist()
        self.labels = [CLASSES.index(label) for label in table["label"]]
        self.patches = np.zeros((len(self.cell_ids), 13, 64, 64), dtype=np.uint16)
        for i, cell_id in enumerate(self.cell_ids):
            with rasterio.open(Path(patches_dir) / f"{cell_id}.tif") as src:
                patch = src.read()  # (13, 64, 64)
            if patch.shape != (13, 64, 64):
                raise ValueError(f"patch {cell_id} has shape {patch.shape}, expected (13, 64, 64)")
            self.patches[i] = patch

        self.bands = bands
        self.mean = mean  # (C,)
        self.std = std  # (C,)
        self.train = train
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.patches[index].astype(np.float32))  # (13, 64, 64)
        if self.bands == "rgb":
            image = select_rgb(image)  # (3, 64, 64)
        image = normalize(image, self.mean, self.std)
        if self.train:
            image = augment(image, self.generator)
        return image, self.labels[index]
