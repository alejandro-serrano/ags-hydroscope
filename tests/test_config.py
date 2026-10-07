"""Config loading and guards on the shared experimental protocol."""

from pathlib import Path

import pytest
import yaml

from hydroscope.config import RUN_KEYS, deep_merge, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RUN_CONFIGS = sorted(p for p in CONFIGS.glob("*.yaml") if p.stem not in {"base", "smoke"})


def test_deep_merge_is_recursive_and_pure():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 3}, "b": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 3}, "b": 4}
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}}


def test_extends_resolves_relative_to_file(tmp_path):
    (tmp_path / "parent.yaml").write_text("seed: 0\noptim: {lr: 0.1, batch_size: 8}\n")
    (tmp_path / "child.yaml").write_text("extends: parent.yaml\noptim: {batch_size: 4}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"seed": 0, "optim": {"lr": 0.1, "batch_size": 4}}


def test_base_matches_protocol():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg["seed"] == 0
    assert cfg["data"]["input_size"] == 224
    assert cfg["data"]["patch_size"] == 64
    assert cfg["data"]["augment"] == {"hflip": True, "vflip": True, "rot90": True}
    assert cfg["optim"] == {"name": "adamw", "lr": 1e-4, "weight_decay": 0.05, "batch_size": 64}
    assert cfg["schedule"] == {"warmup_epochs": 1, "scheduler": "cosine", "max_epochs": 15}
    assert cfg["early_stopping"] == {"monitor": "val_macro_f1", "mode": "max", "patience": 3}


def test_smoke_is_small_and_writes_outside_results():
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert cfg["device"] == "cpu"
    assert cfg["data"]["max_samples"] == 200
    assert cfg["schedule"]["max_epochs"] == 1
    assert not cfg["output"]["results_dir"].startswith("experiments/results")


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=[p.stem for p in RUN_CONFIGS])
def test_run_configs_only_set_run_keys(path):
    raw = yaml.safe_load(path.read_text())
    assert raw.get("extends") == "base.yaml"
    assert set(raw) <= RUN_KEYS, f"{path.name} overrides shared keys: {set(raw) - RUN_KEYS}"
