"""Labels store: upsert, resume, counters, candidate ordering and the cell outline helper."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from shapely.geometry import box

from hydroscope.data import labels as lb
from hydroscope.data.eurosat import CLASSES
from hydroscope.geo.grid import cell_outline_latlon

T1 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
T2 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)


def test_append_creates_header_and_row(tmp_path):
    path = tmp_path / "sub" / "labels.csv"
    frame = lb.upsert_label(path, "r0001_c0001", "Forest", "ana", "b000_000", now=T1)
    assert len(frame) == 1
    assert path.read_text().splitlines()[0] == ",".join(lb.LABEL_COLUMNS)
    row = lb.load_labels(path).iloc[0]
    assert row["label"] == "Forest" and row["split"] == ""
    assert row["timestamp"] == "2026-10-01T12:00:00+00:00"
    assert not list(path.parent.glob("*.tmp"))


def test_same_cell_annotator_replaces(tmp_path):
    path = tmp_path / "labels.csv"
    lb.upsert_label(path, "c1", "Forest", "ana", "b0", now=T1)
    lb.upsert_label(path, "c1", "River", "ana", "b0", now=T2)
    frame = lb.load_labels(path)
    assert len(frame) == 1
    assert frame.iloc[0]["label"] == "River"
    assert frame.iloc[0]["timestamp"].startswith("2026-10-02")


def test_other_annotator_appends(tmp_path):
    path = tmp_path / "labels.csv"
    lb.upsert_label(path, "c1", "Forest", "ana", "b0", now=T1)
    lb.upsert_label(path, "c1", "Forest", "beto", "b0", now=T1)
    assert len(lb.load_labels(path)) == 2


@pytest.mark.parametrize(("label", "annotator"), [("Swamp", "ana"), ("Forest", "  "), ("", "ana")])
def test_invalid_input_raises(tmp_path, label, annotator):
    path = tmp_path / "labels.csv"
    with pytest.raises(ValueError):
        lb.upsert_label(path, "c1", label, annotator, "b0")
    assert not path.exists()


def cands(ids):
    return pd.DataFrame({"cell_id": ids, "block_id": ["b0"] * len(ids)})


def test_resume_index(tmp_path):
    path = tmp_path / "labels.csv"
    c = cands(["a", "b", "c"])
    assert lb.resume_index(c, lb.load_labels(path), "ana") == 0
    lb.upsert_label(path, "a", "Forest", "ana", "b0")
    lb.upsert_label(path, "b", lb.SKIP, "ana", "b0")
    lb.upsert_label(path, "a", "Forest", "beto", "b0")
    # reload from disk
    labels = lb.load_labels(path)
    assert lb.resume_index(c, labels, "ana") == 2
    assert lb.resume_index(c, labels, "beto") == 1
    assert lb.resume_index(c, labels, "nobody") == 0
    lb.upsert_label(path, "c", "River", "ana", "b0")
    assert lb.resume_index(c, lb.load_labels(path), "ana") == 3


def test_class_counts(tmp_path):
    path = tmp_path / "labels.csv"
    assert lb.class_counts(lb.load_labels(path)) == dict.fromkeys(CLASSES, 0)
    lb.upsert_label(path, "a", "Forest", "ana", "b0")
    lb.upsert_label(path, "b", "Forest", "ana", "b0")
    lb.upsert_label(path, "c", lb.SKIP, "ana", "b0")
    counts = lb.class_counts(lb.load_labels(path))
    assert list(counts) == list(CLASSES)
    assert counts["Forest"] == 2 and sum(counts.values()) == 2


def write_cells(path, n=30):
    ids = [f"r{i:04d}_c0000" for i in range(n)]
    pd.DataFrame(
        {
            "cell_id": ids,
            "block_id": [f"b{i // 8:03d}_000" for i in range(n)],
            "has_patch": [i % 3 != 0 for i in range(n)],
        }
    ).to_csv(path, index=False)
    return ids


def test_load_candidates_from_file(tmp_path):
    cells = tmp_path / "cells.csv"
    ids = write_cells(cells)
    cand = tmp_path / "candidates.csv"
    pd.DataFrame({"cell_id": [ids[5], ids[2], ids[9]]}).to_csv(cand, index=False)
    out = lb.load_candidates(cand, cells)
    assert list(out["cell_id"]) == [ids[5], ids[2], ids[9]]
    assert list(out["block_id"]) == ["b000_000", "b000_000", "b001_000"]


def test_load_candidates_fallback_is_seeded(tmp_path):
    cells = tmp_path / "cells.csv"
    ids = write_cells(cells)
    out = lb.load_candidates(tmp_path / "missing.csv", cells, seed=0)
    expected = [c for i, c in enumerate(ids) if i % 3 != 0]
    assert sorted(out["cell_id"]) == sorted(expected)
    assert list(out["cell_id"]) != expected
    again = lb.load_candidates(tmp_path / "missing.csv", cells, seed=0)
    assert list(again["cell_id"]) == list(out["cell_id"])
    other = lb.load_candidates(tmp_path / "missing.csv", cells, seed=1)
    assert list(other["cell_id"]) != list(out["cell_id"])


def test_cell_outline_latlon():
    west, south = 640 * 1220, 640 * 3800
    pts = cell_outline_latlon(box(west, south, west + 640, south + 640))
    assert len(pts) == 5 and pts[0] == pts[-1]
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    assert 21 < sum(lats) / 5 < 23 and -103 < sum(lons) / 5 < -101
    assert max(lats) - min(lats) < 0.01 and max(lons) - min(lons) < 0.01
