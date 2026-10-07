"""EuroSAT dataset wrapper, .npy cache and band statistics (EuroSAT100 for the dataset tests)."""

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torchgeo.datasets import EuroSAT, EuroSAT100

from hydroscope.data.eurosat import (
    BAND_SETS,
    CLASSES,
    EuroSATDataset,
    cache_paths,
    compute_band_stats,
    save_band_stats,
    train_band_stats,
    write_split_cache,
)

ROOT = Path(__file__).resolve().parents[1]
BANDS13 = tuple(EuroSAT.all_band_names)


@pytest.fixture(scope="session")
def eurosat100_root() -> Path:
    """EuroSAT100 downloaded once to data/eurosat100 (git-ignored)."""
    root = ROOT / "data" / "eurosat100"
    for split in ("train", "val", "test"):
        EuroSAT100(root, split, download=True)
    return root


def _dataset(root, split, bands, cache_dir):
    return EuroSATDataset(split, bands, root=root, cache_dir=cache_dir, base_cls=EuroSAT100)


# --- constants ---------------------------------------------------------------------------------


def test_band_sets_follow_torchgeo_order():
    assert BAND_SETS["ms13"] == BANDS13
    assert BAND_SETS["ms13"][-1] == "B8A"
    assert BAND_SETS["rgb"] == ("B04", "B03", "B02")


# --- dataset (EuroSAT100) ----------------------------------------------------------------------


def test_ms13_sample_shape_dtype_and_label(eurosat100_root, tmp_path):
    ds = _dataset(eurosat100_root, "train", "ms13", tmp_path)
    assert not ds.from_cache
    assert len(ds) == 60
    sample = ds[0]
    assert sample["image"].shape == (13, 64, 64)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.int64 and sample["label"].ndim == 0


def test_rgb_sample_shape(eurosat100_root, tmp_path):
    ds = _dataset(eurosat100_root, "train", "rgb", tmp_path)
    assert ds[0]["image"].shape == (3, 64, 64)


def test_rgb_channels_are_b04_b03_b02_of_ms13(eurosat100_root, tmp_path):
    ms13 = _dataset(eurosat100_root, "val", "ms13", tmp_path)
    rgb = _dataset(eurosat100_root, "val", "rgb", tmp_path)
    for i in (0, 5, len(ms13) - 1):
        idx = [BANDS13.index(b) for b in ("B04", "B03", "B02")]
        assert torch.equal(rgb[i]["image"], ms13[i]["image"][idx])
        assert rgb[i]["label"] == ms13[i]["label"]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_labels_in_range_and_classes_match_torchgeo(eurosat100_root, tmp_path, split):
    ds = _dataset(eurosat100_root, split, "rgb", tmp_path)
    labels = [int(ds[i]["label"]) for i in range(len(ds))]
    assert min(labels) >= 0 and max(labels) <= 9
    assert tuple(ds.classes) == CLASSES
    assert tuple(EuroSAT100(eurosat100_root, split).classes) == CLASSES


def test_invalid_split_or_bands_raise(tmp_path):
    with pytest.raises(ValueError):
        EuroSATDataset("trainval", root=tmp_path, download=False)
    with pytest.raises(ValueError):
        EuroSATDataset("train", "nir", root=tmp_path, download=False)


@pytest.mark.parametrize("bands", ["ms13", "rgb"])
def test_cached_sample_identical_to_torchgeo(eurosat100_root, tmp_path, bands):
    cache = tmp_path / "cache"
    x_path, y_path = write_split_cache("train", eurosat100_root, cache, EuroSAT100)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    assert x.dtype == np.uint16 and x.shape == (60, 13, 64, 64)
    assert y.shape == (60,)
    assert not list(cache.glob("*.tmp"))

    cached = _dataset(eurosat100_root, "train", bands, cache)
    direct = _dataset(eurosat100_root, "train", bands, tmp_path / "no_cache")
    assert cached.from_cache and not direct.from_cache
    assert len(cached) == len(direct)
    for i in range(len(direct)):
        a, b = cached[i], direct[i]
        assert a["image"].dtype == torch.float32
        assert torch.equal(a["image"], b["image"])
        assert a["label"] == b["label"]


def test_cached_dataset_pickles_without_memmap(eurosat100_root, tmp_path):
    write_split_cache("test", eurosat100_root, tmp_path, EuroSAT100)
    ds = _dataset(eurosat100_root, "test", "rgb", tmp_path)
    first = ds[0]  # opens the memory maps
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._x is None and clone._y is None
    assert torch.equal(clone[0]["image"], first["image"])


def test_default_cache_dir_is_under_root(eurosat100_root):
    ds = EuroSATDataset("train", root=eurosat100_root, download=False, base_cls=EuroSAT100)
    assert ds.cache_dir == eurosat100_root / "cache"
    x_path, y_path = cache_paths(ds.cache_dir, "train")
    assert x_path.name == "train_x.npy" and y_path.name == "train_y.npy"


def test_write_split_cache_rejects_lossy_values(tmp_path):
    class FractionalEuroSAT:
        def __init__(self, root, split, bands, download):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"image": torch.full((13, 64, 64), 0.5), "label": torch.tensor(0)}

    with pytest.raises(ValueError, match="lossy"):
        write_split_cache("train", tmp_path, tmp_path / "cache", FractionalEuroSAT)
    assert not (tmp_path / "cache" / "train_x.npy").exists()


# --- band statistics ---------------------------------------------------------------------------


def _synthetic(seed=0, n=40, c=3, offset=10_000.0):
    rng = np.random.default_rng(seed)
    scale = np.arange(1, c + 1, dtype=np.float64)[None, :, None, None] * 50
    return rng.normal(offset, 1.0, size=(n, c, 8, 8)) * scale


def test_compute_band_stats_matches_numpy():
    data = _synthetic()
    names = ["a", "b", "c"]
    stats = compute_band_stats([data[:13], data[13:30], data[30:]], names)
    assert list(stats) == names
    for i, name in enumerate(names):
        assert stats[name]["mean"] == pytest.approx(data[:, i].mean(), rel=1e-12)
        assert stats[name]["std"] == pytest.approx(data[:, i].std(), rel=1e-10)


@pytest.mark.parametrize("size", [1, 7, 40])
def test_compute_band_stats_is_chunking_invariant(size):
    data = _synthetic(seed=1)
    names = ["a", "b", "c"]
    reference = compute_band_stats([data], names)
    chunked = compute_band_stats((data[i : i + size] for i in range(0, len(data), size)), names)
    for name in names:
        assert chunked[name]["mean"] == pytest.approx(reference[name]["mean"], rel=1e-12)
        assert chunked[name]["std"] == pytest.approx(reference[name]["std"], rel=1e-10)


def test_compute_band_stats_accepts_tensors_and_validates():
    data = torch.from_numpy(_synthetic(c=2)).float()
    stats = compute_band_stats([data], ["x", "y"])
    assert stats["x"]["mean"] == pytest.approx(float(data[:, 0].double().mean()), rel=1e-9)
    with pytest.raises(ValueError):
        compute_band_stats([data], ["x", "y", "z"])
    with pytest.raises(ValueError):
        compute_band_stats([], ["x"])


def test_train_band_stats_ignores_val_and_test(tmp_path):
    rng = np.random.default_rng(0)
    train = rng.integers(0, 5000, size=(37, 13, 4, 4), dtype=np.uint16)
    for split, high in (("train", 5000), ("val", 60000), ("test", 60000)):
        values = train if split == "train" else rng.integers(50000, high, size=train.shape)
        x_path, y_path = cache_paths(tmp_path, split)
        np.save(x_path, values.astype(np.uint16))
        np.save(y_path, np.zeros(len(values), dtype=np.uint16))
    stats = train_band_stats(tmp_path, chunk=10)
    assert list(stats) == list(BANDS13)
    ref = train.astype(np.float64)
    for i, band in enumerate(BANDS13):
        assert stats[band]["mean"] == pytest.approx(ref[:, i].mean(), rel=1e-12)
        assert stats[band]["std"] == pytest.approx(ref[:, i].std(), rel=1e-10)
    # Same result with the val/test caches removed: they are never read.
    for split in ("val", "test"):
        for path in cache_paths(tmp_path, split):
            path.unlink()
    again = train_band_stats(tmp_path, chunk=512)
    for band in BANDS13:
        assert again[band]["mean"] == pytest.approx(stats[band]["mean"], rel=1e-12)
        assert again[band]["std"] == pytest.approx(stats[band]["std"], rel=1e-10)


def test_save_band_stats_roundtrip(tmp_path):
    stats = {"B01": {"mean": 1.5, "std": 0.5}, "B8A": {"mean": 2.0, "std": 1.0}}
    path = tmp_path / "nested" / "stats.json"
    save_band_stats(stats, path)
    assert json.loads(path.read_text()) == stats


# --- data card (scripts/prepare_eurosat.py) ----------------------------------------------------


def test_render_data_card():
    spec = importlib.util.spec_from_file_location(
        "prepare_eurosat", ROOT / "scripts" / "prepare_eurosat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = {
        "train": np.arange(10),
        "val": np.arange(10) + 1,
        "test": np.arange(10) + 2,
    }
    stats = {b: {"mean": 1.0, "std": 2.0} for b in BANDS13}
    text = module.render_data_card(counts, CLASSES, BANDS13, stats)
    assert text.startswith(module.HEADER)
    assert "| 9 | SeaLake | 9 | 10 | 11 | 30 |" in text
    assert f"| **Total** | {45} | {55} | {65} | {165} |" in text
    assert "| 12 | B8A |" in text
