"""Result JSON contract validation."""

import pytest

from hydroscope.contracts import ContractError, validate_result


@pytest.fixture
def valid_result():
    # Synthetic values: shape of a record only, not a real metric.
    return {
        "run_id": "resnet50_rgb_imagenet",
        "model": "resnet50",
        "bands": "rgb",
        "init": "imagenet",
        "split": "eurosat_test",
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "f1_per_class": {"AnnualCrop": 0.5, "Forest": 0.5},
        "confusion": [[1, 1], [1, 1]],
        "params": 1000,
        "train_time_s": 1.0,
        "peak_mem_mb": None,
        "epochs_to_95": 1,
        "patches_per_s": 1.0,
        "device": "cpu",
        "micro_batch": 32,
        "accum_steps": 2,
        "protocol_version": "1.0",
        "git_commit": "0000000",
    }


def test_valid_result_passes(valid_result):
    assert validate_result(valid_result) is None


def test_invalid_result_raises(valid_result):
    del valid_result["macro_f1"]
    valid_result["params"] = "1000"
    with pytest.raises(ContractError) as excinfo:
        validate_result(valid_result)
    assert "macro_f1" in str(excinfo.value)
    assert "params" in str(excinfo.value)
