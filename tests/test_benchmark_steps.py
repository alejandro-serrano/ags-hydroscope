"""Pure parts of scripts/benchmark_steps.py (no MPS needed)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_steps.py"
spec = importlib.util.spec_from_file_location("benchmark_steps", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def make_row(run_id: str, init: str, h_run: float) -> dict:
    return {
        "run_id": run_id,
        "init": init,
        "params_m": 25.6,
        "s_step": 1.0,
        "s_val_batch": 0.5,
        "peak_mb": 1000.0,
        "min_epoch": 10.0,
        "h_run": h_run,
    }


def test_estimate_arithmetic():
    # 16200 / 64 -> 254 steps, 5400 / 32 -> 169 val batches
    est = bench.estimate(s_step=1.0, s_val=0.5, batch=64, val_batch=32, epochs=15)
    epoch_s = 254 * 1.0 + 169 * 0.5
    assert est["min_epoch"] == pytest.approx(epoch_s / 60)
    assert est["h_run"] == pytest.approx((15 * epoch_s + 2 * 169 * 0.5) / 3600)


def test_replace_generated_block_only_touches_block():
    text = f"intro\n{bench.BEGIN}\nold\n{bench.END}\noutro\n"
    new = bench.replace_generated_block(text, f"{bench.BEGIN}\nnew\n{bench.END}")
    assert new == f"intro\n{bench.BEGIN}\nnew\n{bench.END}\noutro\n"


def test_replace_generated_block_without_markers_raises():
    with pytest.raises(ValueError):
        bench.replace_generated_block("no markers here", "block")


def test_rows_to_markdown_has_header_separator_and_one_line_per_row():
    rows = [make_row("a", "imagenet", 1.0), make_row("b", "ssl4eo", 2.0)]
    lines = bench.rows_to_markdown(rows).split("\n")
    assert len(lines) == 4
    assert lines[0].count("|") == lines[2].count("|")


def test_total_hours_sums_by_init():
    rows = [
        make_row("a", "imagenet", 1.0),
        make_row("b", "imagenet", 2.0),
        make_row("c", "ssl4eo", 4.0),
    ]
    assert bench.total_hours(rows, "imagenet") == pytest.approx(3.0)
    assert bench.total_hours(rows, "ssl4eo") == pytest.approx(4.0)
