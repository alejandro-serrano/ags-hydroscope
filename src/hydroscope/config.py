"""YAML run configs. A run config says ``extends: base.yaml`` and overrides a few keys."""

from copy import deepcopy
from pathlib import Path

import yaml

RUN_KEYS = {"extends", "run_id", "model", "bands", "init"}
"""Keys a run config may set; everything else belongs to the shared protocol in base.yaml."""


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` merged into ``base``, one nested level at a time.

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


def load_config(path: str | Path) -> dict:
    """Load a YAML config. ``extends`` names a parent file in the same folder.

    The returned dict is fully resolved and has no ``extends`` key.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return deep_merge(load_config(path.parent / parent), raw)
