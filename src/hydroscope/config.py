"""YAML run configs with single-parent inheritance via an ``extends`` key."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

RUN_KEYS = frozenset({"extends", "run_id", "model", "bands", "init"})
"""Keys a run config may set; everything else belongs to the shared protocol in base.yaml."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged recursively into ``base``.

    Nested dicts are merged key by key; any other value in ``override`` replaces the base value.
    Neither input is modified.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``extends`` relative to the file's directory.

    The returned dict is fully resolved and has no ``extends`` key.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return deep_merge(load_config(path.parent / parent), raw)
