"""Spatial split of the Aguascalientes labels: blocks stay whole, same seed gives same split."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("split_ags", ROOT / "scripts/split_ags.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def tiny_labels():
    """12 blocks, 3 cells each: two classes and one skip per block."""
    rows = []
    for b in range(12):
        for k, label in enumerate(["Forest", "AnnualCrop", "skip"]):
            rows.append(
                {
                    "cell_id": f"c{b}_{k}",
                    "label": label,
                    "annotator": "A1",
                    "block_id": f"b{b:03d}",
                    "split": "",
                    "timestamp": "",
                }
            )
    return pd.DataFrame(rows)


def test_no_block_in_two_splits():
    labels = tiny_labels()
    block_split, _ = sp.choose_split(labels, n_trials=50, seed=0)
    out = sp.apply_split(labels, block_split)
    splits_per_block = out.groupby("block_id")["split"].nunique()
    assert (splits_per_block == 1).all()


def test_no_cell_in_two_splits():
    labels = tiny_labels()
    block_split, _ = sp.choose_split(labels, n_trials=50, seed=0)
    out = sp.apply_split(labels, block_split)
    assert out["cell_id"].is_unique
    assert set(out["split"]) == {"finetune", "test"}


def test_blocks_are_split_half_and_half():
    block_split, _ = sp.choose_split(tiny_labels(), n_trials=50, seed=0)
    values = list(block_split.values())
    assert values.count("finetune") == 6 and values.count("test") == 6


def test_same_seed_gives_same_split():
    first = sp.choose_split(tiny_labels(), n_trials=50, seed=0)
    second = sp.choose_split(tiny_labels(), n_trials=50, seed=0)
    assert first == second


def test_skip_rows_take_the_split_of_their_block():
    labels = tiny_labels()
    block_split, _ = sp.choose_split(labels, n_trials=50, seed=0)
    out = sp.apply_split(labels, block_split)
    skips = out[out["label"] == "skip"]
    for _, row in skips.iterrows():
        assert row["split"] == block_split[row["block_id"]]


def test_class_table_ignores_skip_and_flags_small_test():
    labels = tiny_labels()
    block_split, _ = sp.choose_split(labels, n_trials=50, seed=0)
    table = sp.class_table(sp.apply_split(labels, block_split))
    forest = table[table["class"] == "Forest"].iloc[0]
    assert forest["total"] == 12
    assert forest["finetune"] + forest["test"] == 12
    assert forest["flag"] == "< 10 in test"
