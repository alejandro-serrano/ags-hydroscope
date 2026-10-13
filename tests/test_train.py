"""Learning-rate schedule, gradient accumulation, metrics, rgb selection and the smoke pipeline."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hydroscope.config import load_config
from hydroscope.contracts import validate_result
from hydroscope.data.eurosat import EuroSATDataset
from hydroscope.data.transforms import select_rgb
from hydroscope.training.evaluate import compute_metrics
from hydroscope.training.train import lr_at_step, run_epoch, run_training

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "eurosat" / "cache"
HAS_CACHE = (CACHE_DIR / "test_x.npy").exists() and (CACHE_DIR / "train_x.npy").exists()


def test_lr_reaches_base_lr_at_end_of_warmup():
    assert lr_at_step(9, total_steps=100, warmup_steps=10, base_lr=1e-4) == pytest.approx(1e-4)
    assert lr_at_step(0, total_steps=100, warmup_steps=10, base_lr=1e-4) == pytest.approx(1e-5)


def test_lr_is_cosine_after_warmup_and_zero_at_the_end():
    assert lr_at_step(10, total_steps=100, warmup_steps=10, base_lr=1e-4) == pytest.approx(1e-4)
    assert lr_at_step(55, total_steps=100, warmup_steps=10, base_lr=1e-4) == pytest.approx(5e-5)
    assert lr_at_step(100, total_steps=100, warmup_steps=10, base_lr=1e-4) == pytest.approx(0.0)
    assert lr_at_step(99, total_steps=100, warmup_steps=10, base_lr=1e-4) < 1e-7


def make_small_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))  # no BatchNorm


def one_sgd_step(micro_batch: int, accum_steps: int) -> torch.Tensor:
    # Returns the weights after one optimizer step over 64 samples.
    torch.manual_seed(1)
    x = torch.randn(64, 3, 8, 8)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=micro_batch, shuffle=False)
    model = make_small_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = {
        "step": 0,
        "total_steps": 10,
        "warmup_steps": 1,
        "base_lr": 0.1,
        "max_steps": None,
        "stop": False,
        "tracker": FakeTracker(),
    }
    criterion = nn.CrossEntropyLoss()
    run_epoch(model, loader, criterion, torch.device("cpu"), 8, optimizer, accum_steps, state)
    assert state["step"] == 1
    return model[1].weight.detach().clone()


class FakeTracker:
    def update(self):
        pass


def test_accumulating_2_micro_batches_of_32_equals_one_batch_of_64():
    weights_accumulated = one_sgd_step(micro_batch=32, accum_steps=2)
    weights_big_batch = one_sgd_step(micro_batch=64, accum_steps=1)
    assert torch.allclose(weights_accumulated, weights_big_batch, atol=1e-6)


def test_compute_metrics_on_a_toy_example():
    labels = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 2, 0]
    metrics = compute_metrics(labels, preds)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["f1_per_class"]["AnnualCrop"] == pytest.approx(0.5)
    assert metrics["f1_per_class"]["Forest"] == pytest.approx(0.8)
    assert metrics["f1_per_class"]["HerbaceousVegetation"] == pytest.approx(2 / 3)
    assert metrics["f1_per_class"]["Highway"] == 0.0
    assert len(metrics["confusion"]) == 10 and len(metrics["confusion"][0]) == 10
    assert metrics["confusion"][0][1] == 1  # one true AnnualCrop predicted as Forest
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 10)


@pytest.mark.skipif(not HAS_CACHE, reason="EuroSAT .npy cache not found")
def test_rgb_dataset_equals_select_rgb_of_ms13():
    ms13 = EuroSATDataset("val", bands="ms13", cache_dir=CACHE_DIR)[0]["image"]
    rgb = EuroSATDataset("val", bands="rgb", cache_dir=CACHE_DIR)[0]["image"]
    assert torch.equal(rgb, select_rgb(ms13))


@pytest.mark.skipif(not HAS_CACHE, reason="EuroSAT .npy cache not found")
def test_smoke_pipeline_writes_a_valid_result_json(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["data"]["max_samples"] = 16
    cfg["data"]["stats"] = str(ROOT / cfg["data"]["stats"])
    cfg["data"]["cache_dir"] = str(CACHE_DIR)
    cfg["optim"]["batch_size"] = 8
    cfg["optim"]["micro_batch_size"] = 4
    cfg["output"] = {
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "results_dir": str(tmp_path / "results"),
        "logs_dir": str(tmp_path / "logs"),
    }
    run_training(cfg)
    result = json.loads((tmp_path / "results" / "smoke_eurosat_test.json").read_text())
    validate_result(result)
    assert result["run_id"] == "smoke_eurosat_test"
    assert result["accum_steps"] == 2
    assert (tmp_path / "ckpt" / "smoke.pt").exists()
    assert (tmp_path / "logs" / "smoke.csv").exists()
