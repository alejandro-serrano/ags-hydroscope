"""Validation of the data contracts in docs/contracts.md."""

from __future__ import annotations

from typing import Any

RESULT_FIELDS: dict[str, str] = {
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
    "protocol_version": "str",
    "git_commit": "str",
}
"""Result JSON fields (contract 4) and their expected types."""

NULLABLE_RESULT_FIELDS = frozenset({"train_time_s", "peak_mem_mb", "epochs_to_95"})
"""Result fields that may be ``None`` when they do not apply."""


class ContractError(ValueError):
    """A record does not satisfy its data contract."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


_CHECKS = {
    "str": lambda v: isinstance(v, str),
    "int": _is_int,
    "float": _is_float,
    "dict[str, float]": lambda v: (
        isinstance(v, dict) and all(isinstance(k, str) and _is_float(x) for k, x in v.items())
    ),
    "list[list[int]]": lambda v: (
        isinstance(v, list) and all(isinstance(row, list) and all(map(_is_int, row)) for row in v)
    ),
}


def validate_result(result: dict[str, Any]) -> None:
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
        if not _CHECKS[expected](value):
            problems.append(f"field '{field}' must be {expected}, got {type(value).__name__}")
    if problems:
        raise ContractError("invalid result: " + "; ".join(problems))
