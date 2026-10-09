"""Labeling store for Aguascalientes cells (contract 2 in ``docs/contracts.md``).

Pure helpers used by ``labeling/label.ipynb``: candidate ordering, atomic upsert of one answer per
``(cell_id, annotator)``, resume position and per-class counters. ``split`` is left empty here;
the spatial split script fills it later.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hydroscope.data.eurosat import CLASSES

LABEL_COLUMNS = ["cell_id", "label", "annotator", "block_id", "split", "timestamp"]
SKIP = "skip"
VALID_LABELS: tuple[str, ...] = (*CLASSES, SKIP)
TARGET_PER_CLASS = 50
SEED = 0


def load_candidates(
    candidates_csv: str | Path, cells_csv: str | Path, seed: int = SEED
) -> pd.DataFrame:
    """Ordered cells to label.

    Uses ``candidates_csv`` (file order kept) when it exists; ``block_id`` is joined from
    ``cells_csv`` if missing. Otherwise the ``has_patch`` rows of ``cells_csv`` are shuffled with
    ``numpy.random.default_rng(seed)``.

    Returns:
        DataFrame with columns ``cell_id`` and ``block_id`` and a fresh RangeIndex.
    """
    candidates_csv, cells_csv = Path(candidates_csv), Path(cells_csv)
    cells = pd.read_csv(cells_csv, dtype={"cell_id": str, "block_id": str})
    if candidates_csv.exists():
        cand = pd.read_csv(candidates_csv, dtype={"cell_id": str, "block_id": str})
        if "cell_id" not in cand.columns:
            raise ValueError(f"{candidates_csv} must have a 'cell_id' column")
        if "block_id" not in cand.columns:
            cand = cand[["cell_id"]].merge(cells[["cell_id", "block_id"]], on="cell_id", how="left")
        return cand[["cell_id", "block_id"]].reset_index(drop=True)
    has_patch = cells["has_patch"].astype(str).str.lower() == "true"
    subset = cells.loc[has_patch, ["cell_id", "block_id"]].reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(subset))
    return subset.iloc[order].reset_index(drop=True)


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read ``labels.csv`` as strings (NaN -> ""); empty frame with ``LABEL_COLUMNS`` if missing."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype=str) for c in LABEL_COLUMNS})
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    for col in LABEL_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    return frame[LABEL_COLUMNS]


def upsert_label(
    path: str | Path,
    cell_id: str,
    label: str,
    annotator: str,
    block_id: str,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Write one answer, replacing any earlier one for the same ``(cell_id, annotator)``.

    Args:
        path: ``labels.csv`` location (created if missing).
        label: one of ``VALID_LABELS``.
        annotator: non-empty (after stripping) annotator id.
        now: timestamp override for tests; defaults to current UTC time.

    Returns:
        The updated labels frame. The file is written atomically.

    Raises:
        ValueError: invalid label or empty annotator.
    """
    if label not in VALID_LABELS:
        raise ValueError(f"invalid label {label!r}; expected one of {VALID_LABELS}")
    annotator = annotator.strip()
    if not annotator:
        raise ValueError("annotator must be non-empty")
    path = Path(path)
    stamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    row = {
        "cell_id": cell_id,
        "label": label,
        "annotator": annotator,
        "block_id": block_id,
        "split": "",
        "timestamp": stamp,
    }
    frame = load_labels(path)
    same = (frame["cell_id"] == cell_id) & (frame["annotator"] == annotator)
    if same.any():
        first = same.idxmax()
        for key, value in row.items():
            frame.loc[first, key] = value
        frame = frame[~same | (frame.index == first)].reset_index(drop=True)
    else:
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    _atomic_write(frame, path)
    return frame


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` via a temp file in the same directory and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def resume_index(candidates: pd.DataFrame, labels: pd.DataFrame, annotator: str) -> int:
    """Index of the first candidate not answered by ``annotator`` (``skip`` counts as answered).

    Returns ``len(candidates)`` when every candidate is answered.
    """
    done = set(labels.loc[labels["annotator"] == annotator.strip(), "cell_id"])
    for i, cid in enumerate(candidates["cell_id"]):
        if cid not in done:
            return i
    return len(candidates)


def class_counts(labels: pd.DataFrame) -> dict[str, int]:
    """Rows per class over all 10 ``CLASSES`` (zeros included); ``skip`` excluded."""
    counts = labels["label"].value_counts() if len(labels) else pd.Series(dtype=int)
    return {c: int(counts.get(c, 0)) for c in CLASSES}
