"""Gradient accumulation, on-device resize, DataLoader flags and CLI of the shared loop."""

import pytest
import torch
from torch.utils.data import TensorDataset

from hydroscope.training.loop_utils import (
    accumulation_steps,
    build_arg_parser,
    make_dataloader,
    resize_batch,
)


def test_accumulation_steps():
    assert accumulation_steps(64, 32) == 2
    assert accumulation_steps(16, 8) == 2
    assert accumulation_steps(64, 64) == 1


@pytest.mark.parametrize(("batch", "micro"), [(64, 24), (64, 128), (64, 0), (0, 8), (64, -32)])
def test_accumulation_steps_invalid(batch, micro):
    with pytest.raises(ValueError):
        accumulation_steps(batch, micro)


def test_resize_batch_shape_and_dtype():
    x = torch.rand(2, 13, 64, 64)
    y = resize_batch(x)
    assert y.shape == (2, 13, 224, 224)
    assert y.dtype == x.dtype
    assert resize_batch(x, size=32).shape == (2, 13, 32, 32)


def test_resize_batch_is_deterministic_and_bilinear():
    x = torch.rand(2, 3, 64, 64)
    assert torch.equal(resize_batch(x), resize_batch(x))
    constant = torch.full((1, 3, 64, 64), 7.0)
    assert torch.allclose(resize_batch(constant), torch.full((1, 3, 224, 224), 7.0))


def test_resize_batch_keeps_device():
    x = torch.rand(1, 3, 64, 64)
    assert resize_batch(x).device == x.device
    if torch.backends.mps.is_available():
        assert resize_batch(x.to("mps")).device.type == "mps"


def _loader(**kwargs):
    ds = TensorDataset(torch.arange(100))
    return make_dataloader(ds, micro_batch_size=10, shuffle=True, **kwargs)


def test_dataloader_flags():
    loader = _loader(num_workers=2)
    assert loader.batch_size == 10
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.pin_memory is False


def test_dataloader_without_workers_has_no_persistent_workers():
    loader = _loader(num_workers=0)
    assert loader.num_workers == 0
    assert loader.persistent_workers is False


def test_dataloader_default_num_workers_is_two():
    assert _loader().num_workers == 2


def test_dataloader_shuffle_is_seeded():
    def order(seed):
        return torch.cat([b[0] for b in _loader(num_workers=0, seed=seed)])

    assert torch.equal(order(0), order(0))
    assert not torch.equal(order(0), order(1))
    assert sorted(order(0).tolist()) == list(range(100))


def test_dataloader_no_shuffle_keeps_order():
    loader = make_dataloader(TensorDataset(torch.arange(20)), 5, shuffle=False, num_workers=0)
    assert torch.equal(torch.cat([b[0] for b in loader]), torch.arange(20))


def test_arg_parser():
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "configs/smoke.yaml"])
    assert args.config == "configs/smoke.yaml"
    assert args.max_steps is None
    args = parser.parse_args(["--config", "x.yaml", "--max-steps", "100"])
    assert args.max_steps == 100
    with pytest.raises(SystemExit):
        parser.parse_args([])
