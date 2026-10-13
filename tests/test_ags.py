"""Aguascalientes dataset, finetune loader and class weights, on tiny GeoTIFFs."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from hydroscope.data.ags import AguascalientesDataset
from hydroscope.data.labels import LABEL_COLUMNS
from hydroscope.data.transforms import select_rgb
from hydroscope.training.finetune import class_weights
from hydroscope.training.loop_utils import make_ags_loader

# (cell_id, label, split)
ROWS = [
    ("r0001_c0001", "Forest", "finetune"),
    ("r0001_c0002", "AnnualCrop", "finetune"),
    ("r0002_c0001", "skip", "finetune"),
    ("r0003_c0001", "River", "test"),
    ("r0003_c0002", "Forest", "test"),
]


def write_patch(path, value):
    data = np.full((13, 64, 64), value, dtype=np.uint16)
    data[:, 0, 0] += 1  # a pixel that differs, so flips and rotations are visible
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=13,
        dtype="uint16",
        crs="EPSG:32613",
        transform=from_origin(0, 640, 10, 10),
    ) as dst:
        dst.write(data)


def make_files(tmp_path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    table = []
    for i, (cell_id, label, split) in enumerate(ROWS):
        write_patch(patches_dir / f"{cell_id}.tif", 1000 + 100 * i)
        table.append([cell_id, label, "A1", "b000_000", split, ""])
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(table, columns=LABEL_COLUMNS).to_csv(labels_csv, index=False)
    return labels_csv, patches_dir


def test_dataset_reads_one_patch_as_normalized_float(tmp_path):
    labels_csv, patches_dir = make_files(tmp_path)
    mean, std = torch.full((13,), 500.0), torch.full((13,), 100.0)
    dataset = AguascalientesDataset("finetune", "ms13", mean, std, False, labels_csv, patches_dir)
    image, label = dataset[0]
    assert image.shape == (13, 64, 64) and image.dtype == torch.float32
    assert label == 1  # Forest
    assert image[0, 5, 5].item() == pytest.approx((1000 - 500) / 100)
    assert image[0, 0, 0].item() == pytest.approx((1001 - 500) / 100)

    rgb = AguascalientesDataset(
        "finetune", "rgb", mean[:3], std[:3], False, labels_csv, patches_dir
    )[0][0]
    assert rgb.shape == (3, 64, 64)
    assert torch.equal(rgb, select_rgb(image))


def test_dataset_keeps_only_its_split_and_ignores_skip(tmp_path):
    labels_csv, patches_dir = make_files(tmp_path)
    mean, std = torch.zeros(13), torch.ones(13)
    finetune = AguascalientesDataset("finetune", "ms13", mean, std, False, labels_csv, patches_dir)
    test = AguascalientesDataset("test", "ms13", mean, std, False, labels_csv, patches_dir)
    assert finetune.cell_ids == ["r0001_c0001", "r0001_c0002"]
    assert test.cell_ids == ["r0003_c0001", "r0003_c0002"]
    with pytest.raises(ValueError):
        AguascalientesDataset("val", "ms13", mean, std, False, labels_csv, patches_dir)


def test_dataset_augmentation_is_deterministic_with_a_fixed_seed(tmp_path):
    labels_csv, patches_dir = make_files(tmp_path)
    mean, std = torch.zeros(13), torch.ones(13)
    first = AguascalientesDataset("finetune", "ms13", mean, std, True, labels_csv, patches_dir)
    second = AguascalientesDataset("finetune", "ms13", mean, std, True, labels_csv, patches_dir)
    for i in range(2):
        assert torch.equal(first[i][0], second[i][0])


def test_finetune_loader_never_contains_test_cell_ids(tmp_path):
    labels_csv, patches_dir = make_files(tmp_path)
    cfg = {"bands": "ms13", "seed": 0, "optim": {"micro_batch_size": 2}}
    mean, std = torch.zeros(13), torch.ones(13)
    loader = make_ags_loader(cfg, "finetune", mean, std, True, labels_csv, patches_dir)
    assert loader.dataset.cell_ids == ["r0001_c0001", "r0001_c0002"]
    images, labels = next(iter(loader))
    assert images.shape == (2, 13, 64, 64) and labels.shape == (2,)


def test_class_weights_are_inverse_frequency():
    labels = [0, 0, 0, 1]
    weights = class_weights(labels, num_classes=2)
    assert weights.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])


def test_class_weights_raise_when_a_class_is_missing():
    with pytest.raises(ValueError):
        class_weights([0, 0, 1], num_classes=3)
