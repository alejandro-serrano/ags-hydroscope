"""Device selection, MPS fallback logging and peak-memory tracking."""

import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import hydroscope
from hydroscope.utils import device as device_mod
from hydroscope.utils.device import PRECISION, PeakMemoryTracker, log_mps_fallbacks, select_device

SRC = str(Path(__file__).resolve().parents[1] / "src")
MB = 1024**2


def test_precision_is_fp32():
    assert PRECISION == "fp32"


def test_select_device_cpu_and_invalid():
    assert select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError):
        select_device("tpu")


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_select_device_auto_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device("auto").type == expected
    assert select_device().type == expected


def test_select_device_unavailable_accelerator_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        select_device("cuda")
    with pytest.raises(RuntimeError):
        select_device("mps")


def _run_python(code: str, **env_overrides) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTORCH_")}
    env.update({"PYTHONPATH": SRC, **env_overrides})
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def test_import_sets_mps_fallback_and_leaves_watermark_alone():
    code = (
        "import os, hydroscope;"
        "print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),"
        " os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO'),"
        " hydroscope.TORCH_IMPORTED_BEFORE_MPS_FALLBACK)"
    )
    assert _run_python(code) == "1 None False"


def test_import_respects_user_value_and_detects_early_torch_import():
    code = "import os, hydroscope; print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])"
    assert _run_python(code, PYTORCH_ENABLE_MPS_FALLBACK="0") == "0"
    code = "import torch, hydroscope; print(hydroscope.TORCH_IMPORTED_BEFORE_MPS_FALLBACK)"
    assert _run_python(code) == "True"


@pytest.fixture
def restore_warning_capture():
    yield
    logging.captureWarnings(False)


def test_log_mps_fallbacks_routes_cpu_fallback_warning_to_logging(
    caplog, monkeypatch, restore_warning_capture
):
    monkeypatch.setattr(hydroscope, "TORCH_IMPORTED_BEFORE_MPS_FALLBACK", False)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    log_mps_fallbacks()
    message = (
        "The operator 'aten::foo' is not currently supported on the MPS backend and will "
        "fall back to run on the CPU."
    )
    with caplog.at_level(logging.WARNING):
        warnings.warn(message, UserWarning, stacklevel=1)
        warnings.warn(message, UserWarning, stacklevel=1)  # repeats are not suppressed
    fallbacks = [r for r in caplog.records if "fall back to run on the CPU" in r.getMessage()]
    assert len(fallbacks) == 2
    assert all(r.levelno == logging.WARNING for r in fallbacks)
    assert not [r for r in caplog.records if r.name == device_mod.__name__]


def test_log_mps_fallbacks_warns_if_env_was_not_set_in_time(
    caplog, monkeypatch, restore_warning_capture
):
    monkeypatch.setattr(hydroscope, "TORCH_IMPORTED_BEFORE_MPS_FALLBACK", True)
    with caplog.at_level(logging.WARNING):
        log_mps_fallbacks()
    assert any(
        r.name == device_mod.__name__ and "PYTORCH_ENABLE_MPS_FALLBACK" in r.getMessage()
        for r in caplog.records
    )


def test_peak_memory_tracker_is_none_on_cpu():
    tracker = PeakMemoryTracker("cpu")
    tracker.update()
    assert tracker.peak_mb is None


def test_peak_memory_tracker_takes_max_of_mps_samples(monkeypatch):
    samples = iter([100 * MB, 300 * MB, 200 * MB])
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: next(samples))
    tracker = PeakMemoryTracker(torch.device("mps"))
    for _ in range(3):
        tracker.update()
    assert tracker._peak_bytes == 300 * MB


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_peak_memory_tracker_on_real_mps():
    tracker = PeakMemoryTracker("mps")
    x = torch.zeros(1024, 1024, device="mps")
    tracker.update()
    first = tracker.peak_mb
    assert first is not None and first > 0
    del x
    tracker.update()
    assert tracker.peak_mb >= first
