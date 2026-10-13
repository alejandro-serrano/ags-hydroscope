"""Pure parts of scripts/make_results.py (no result files needed)."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_results.py"
spec = importlib.util.spec_from_file_location("make_results", SCRIPT)
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)


def make_summary(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Rows are (run_id, ags_last_block, patches_per_s)."""
    data = []
    for run_id, f1, speed in rows:
        data.append(
            {"run_id": run_id, "label": run_id, "ags_last_block": f1, "patches_per_s": speed}
        )
    return pd.DataFrame(data)


def test_eurosat_f1_uses_only_the_macro_classes():
    record = {"f1_per_class": {"A": 1.0, "B": 0.5, "River": 0.0}}
    assert mr.eurosat_f1_on(record, ["A", "B"]) == pytest.approx(0.75)


def test_recovery_formula_and_nan_when_drop_is_not_positive():
    drop_value = mr.drop(0.9, 0.5)
    assert drop_value == pytest.approx(0.4)
    assert mr.recovery(0.7, 0.5, drop_value) == pytest.approx(0.5)
    assert pd.isna(mr.recovery(0.7, 0.5, 0.0))
    assert pd.isna(mr.recovery(0.7, 0.5, -0.1))


def test_selection_picks_highest_last_block_f1():
    summary = make_summary([("a", 0.60, 100.0), ("b", 0.70, 50.0), ("c", 0.50, 300.0)])
    selection = mr.select_model(summary, ["a", "b", "c"])
    assert selection["winner"] == "b"
    assert selection["tie_break_applied"] is False
    assert selection["runner_up"] == "a"


def test_selection_tie_within_0_01_goes_to_higher_patches_per_s():
    summary = make_summary([("a", 0.700, 100.0), ("b", 0.695, 200.0), ("c", 0.600, 900.0)])
    selection = mr.select_model(summary, ["a", "b", "c"])
    assert selection["winner"] == "b"
    assert selection["tie_break_applied"] is True
    assert selection["throughput_difference"] == pytest.approx(100.0)


def test_macro_classes_must_match_across_results():
    def record(classes: list[str]) -> dict:
        part = {"macro_f1_classes": classes}
        return {"ags": part, "head": part, "last_block": part}

    same = {"r1": record(["A", "B"]), "r2": record(["A", "B"])}
    assert mr.macro_classes(same) == ["A", "B"]
    different = {"r1": record(["A", "B"]), "r2": record(["A"])}
    with pytest.raises(ValueError):
        mr.macro_classes(different)
