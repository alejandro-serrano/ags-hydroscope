"""Validation of the result JSON contract (contract 4 in docs/contracts.md)."""

RESULT_FIELDS = {
    "run_id": "str",
    "model": "str",
    "bands": "str",
    "init": "str",
    "split": "str",
    "accuracy": "float",
    "macro_f1": "float",
    "f1_per_class": "dict[str, float]",
    "confusion": "list[list[int]]",
    "params": "int",
    "train_time_s": "float",
    "peak_mem_mb": "float",
    "epochs_to_95": "int",
    "patches_per_s": "float",
    "device": "str",
    "micro_batch": "int",
    "accum_steps": "int",
    "protocol_version": "str",
    "git_commit": "str",
    "git_dirty": "bool",
}
"""Every field a result JSON must have, with the type we expect (as written in the contract)."""

NULLABLE_RESULT_FIELDS = ("train_time_s", "peak_mem_mb", "epochs_to_95")
"""Fields that may be ``None`` when they do not apply (e.g. peak memory on CPU)."""


class ContractError(ValueError):
    """A record does not satisfy its data contract."""


def is_int(value) -> bool:
    # bool is a subclass of int in Python, so True/False must be rejected explicitly.
    return isinstance(value, int) and not isinstance(value, bool)


def is_float(value) -> bool:
    # An int is accepted where a float is expected (JSON writes 1.0 as 1).
    return isinstance(value, int | float) and not isinstance(value, bool)


def has_expected_type(value, expected: str) -> bool:
    """True if ``value`` matches one of the type names used in ``RESULT_FIELDS``."""
    if expected == "str":
        return isinstance(value, str)
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "int":
        return is_int(value)
    if expected == "float":
        return is_float(value)
    if expected == "dict[str, float]":
        if not isinstance(value, dict):
            return False
        for key, item in value.items():
            if not isinstance(key, str) or not is_float(item):
                return False
        return True
    if expected == "list[list[int]]":
        if not isinstance(value, list):
            return False
        for row in value:
            if not isinstance(row, list):
                return False
            for item in row:
                if not is_int(item):
                    return False
        return True
    raise ValueError(f"unknown type name in RESULT_FIELDS: {expected!r}")


def validate_result(result: dict) -> None:
    """Check a result record against contract 4 of docs/contracts.md.

    Every field must be present with the expected type; fields in ``NULLABLE_RESULT_FIELDS`` may
    be ``None``. Extra fields are allowed.

    Raises:
        ContractError: listing every missing or mistyped field.
    """
    if not isinstance(result, dict):
        raise ContractError(f"result must be a dict, got {type(result).__name__}")
    problems = []
    for field, expected in RESULT_FIELDS.items():
        if field not in result:
            problems.append(f"missing field '{field}'")
            continue
        value = result[field]
        if value is None and field in NULLABLE_RESULT_FIELDS:
            continue
        if not has_expected_type(value, expected):
            problems.append(f"field '{field}' must be {expected}, got {type(value).__name__}")
    if problems:
        raise ContractError("invalid result: " + "; ".join(problems))
