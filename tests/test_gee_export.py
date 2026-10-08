"""Pure helpers and CLI wiring of the Earth Engine export (no network, no Earth Engine)."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest

from hydroscope.data.eurosat import BAND_SETS
from hydroscope.geo import gee_export as gx

# ---------------------------------------------------------------------------------------------
# Band mapping and QA60
# ---------------------------------------------------------------------------------------------


def test_band_mapping_matches_eurosat_order():
    assert tuple(gx.eurosat_band_names()) == BAND_SETS["ms13"]
    assert len(gx.gee_band_names()) == 13
    assert gx.gee_band_names()[-1] == "B8A" and gx.eurosat_band_names()[-1] == "B8A"
    assert gx.gee_band_names()[:3] == ["B1", "B2", "B3"]
    assert gx.band_mapping()["B1"] == "B01" and gx.band_mapping()["B12"] == "B12"
    assert len(set(gx.gee_band_names())) == 13


def test_export_bands_end_with_valid_obs():
    names = gx.export_band_names()
    assert len(names) == 14 and names[-1] == "valid_obs"
    assert names[:13] == list(BAND_SETS["ms13"])


def test_qa60_bits():
    assert gx.QA60_OPAQUE_MASK == 1 << 10 == 1024
    assert gx.QA60_CIRRUS_MASK == 1 << 11 == 2048
    assert gx.QA60_CLOUD_MASK == 3072


@pytest.mark.parametrize(
    ("qa", "cloudy"),
    [(0, False), (1 << 10, True), (1 << 11, True), ((1 << 10) | (1 << 11), True), (1 << 9, False)],
)
def test_is_cloudy(qa, cloudy):
    assert gx.is_cloudy(qa) is cloudy


def test_qa60_noop_warning_rule():
    assert gx.qa60_mask_is_noop(0.0, 12.0) is True
    assert gx.qa60_mask_is_noop(0.0, 0.5) is False  # almost cloud-free month
    assert gx.qa60_mask_is_noop(0.03, 12.0) is False
    assert gx.qa60_mask_is_noop(None, 12.0) is False
    assert gx.qa60_mask_is_noop(0.0, None) is False


# ---------------------------------------------------------------------------------------------
# 640 m snapping and crsTransform
# ---------------------------------------------------------------------------------------------


def test_snap_origin_is_multiple_of_grid_and_covers_bbox():
    x0, y0 = gx.snap_origin(783_456.7, 2_456_789.1)
    assert x0 % 640 == 0 and y0 % 640 == 0
    assert x0 <= 783_456.7 < x0 + 640
    assert y0 >= 2_456_789.1 > y0 - 640


def test_snap_origin_keeps_lattice_points_and_handles_negatives():
    assert gx.snap_origin(1280.0, 640.0) == (1280, 640)
    assert gx.snap_origin(-1.0, -1.0) == (-640, 0)
    assert gx.snap_origin(-641.0, -640.0) == (-1280, -640)


def test_snap_origin_rejects_incompatible_grid():
    with pytest.raises(ValueError):
        gx.snap_origin(0, 0, grid=645, pixel=10)


def test_snap_extent_contains_bounds_and_is_aligned():
    bounds = (700_123.4, 2_400_050.9, 760_999.9, 2_470_321.0)
    x0, y0, x1, y1 = gx.snap_extent(bounds)
    assert all(v % 640 == 0 for v in (x0, y0, x1, y1))
    assert x0 <= bounds[0] and x1 >= bounds[2]
    assert y1 <= bounds[1] and y0 >= bounds[3]
    width, height = gx.raster_size((x0, y0, x1, y1))
    assert width % 64 == 0 and height % 64 == 0


def test_snap_extent_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        gx.snap_extent((10, 0, 0, 10))


def test_ring_extent_in_metres():
    ring = [[700_000, 2_400_000], [760_000, 2_400_000], [760_000, 2_470_000], [700_000, 2_470_000]]
    assert gx.ring_extent(ring) == (700_000, 2_400_000, 760_000, 2_470_000)


@pytest.mark.parametrize("ring", [[], [[-102.9, 21.6], [-101.8, 21.6], [-101.8, 22.5]]])
def test_ring_extent_rejects_empty_or_degrees(ring):
    with pytest.raises(ValueError):
        gx.ring_extent(ring)


def test_state_bounds_utm_uses_projected_bounds_with_error_margin():
    # Regression: bounds() of a reprojected geometry with zero error margin fails on the server.
    geometry = MagicMock()
    box = geometry.bounds.return_value
    box.coordinates.return_value.getInfo.return_value = [
        [[700_000, 2_400_000], [760_000, 2_400_000], [760_000, 2_470_000], [700_000, 2_470_000]]
    ]
    assert gx.state_bounds_utm(MagicMock(), geometry) == (700_000, 2_400_000, 760_000, 2_470_000)
    kwargs = geometry.bounds.call_args.kwargs
    assert kwargs["proj"] == "EPSG:32613" and kwargs["maxError"] > 0
    geometry.transform.assert_not_called()


def test_crs_transform_shape():
    assert gx.crs_transform(512_000, 2_400_640) == [10, 0, 512_000, 0, -10, 2_400_640]


def test_export_parameters():
    params = gx.export_parameters((512_000, 2_400_640, 512_640, 2_400_000))
    assert params["crs"] == "EPSG:32613"
    assert params["crsTransform"] == [10, 0, 512_000, 0, -10, 2_400_640]
    assert params["folder"] == "ags-hydroscope" and params["fileNamePrefix"] == "s2_ags_2024_04"
    assert (params["width_px"], params["height_px"]) == (64, 64)
    assert params["maxPixels"] == 10**10
    assert len(params["bands"]) == 14


# ---------------------------------------------------------------------------------------------
# Environment and laziness
# ---------------------------------------------------------------------------------------------


def test_get_project_missing_or_empty():
    with pytest.raises(gx.MissingProjectError, match="EE_PROJECT"):
        gx.get_project({})
    with pytest.raises(gx.MissingProjectError):
        gx.get_project({"EE_PROJECT": "  "})
    assert gx.get_project({"EE_PROJECT": "my-project"}) == "my-project"


def test_module_imports_without_ee(monkeypatch):
    monkeypatch.setitem(sys.modules, "ee", None)  # any `import ee` now raises ImportError
    monkeypatch.delitem(sys.modules, "hydroscope.geo.gee_export", raising=False)
    module = importlib.import_module("hydroscope.geo.gee_export")
    assert module.snap_origin(1, 1) == (0, 640)
    with pytest.raises(ImportError):
        module.initialize_ee("any")


def test_main_without_project_exits_2_and_never_touches_ee(monkeypatch, capsys):
    monkeypatch.delenv("EE_PROJECT", raising=False)

    def boom(project):
        raise AssertionError("ee must not be initialized without a project")

    monkeypatch.setattr(gx, "initialize_ee", boom)
    assert gx.main(["--dry-run"]) == 2
    assert "EE_PROJECT" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# Boundary and CLI wiring with a mocked ee
# ---------------------------------------------------------------------------------------------


def _mock_states(count, names):
    ee = MagicMock()
    states = ee.FeatureCollection.return_value.filter.return_value
    selected = states.filter.return_value
    selected.size.return_value.getInfo.return_value = count
    states.aggregate_array.return_value.distinct.return_value.getInfo.return_value = names
    return ee, selected


def test_state_geometry_not_unique_lists_sorted_names_and_exits(capsys):
    ee, _ = _mock_states(0, ["Zacatecas", "Jalisco", "Aguascalientes "])
    with pytest.raises(SystemExit) as exc:
        gx.get_state_geometry(ee)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.index("Aguascalientes ") < err.index("Jalisco") < err.index("Zacatecas")


def test_state_geometry_unique_returns_geometry():
    ee, selected = _mock_states(1, [])
    assert gx.get_state_geometry(ee) is selected.geometry.return_value


def _patch_pipeline(monkeypatch):
    ee = MagicMock()
    monkeypatch.setenv("EE_PROJECT", "dummy-test-project")
    monkeypatch.setattr(gx, "initialize_ee", lambda project: ee)
    monkeypatch.setattr(gx, "get_state_geometry", lambda ee_, boundary=None: MagicMock())
    monkeypatch.setattr(gx, "filtered_collection", lambda ee_, geom: MagicMock())
    monkeypatch.setattr(
        gx,
        "collection_diagnostics",
        lambda ee_, col, geom: {
            "scenes": 12,
            "mean_cloud_percent": 8.0,
            "qa60_flagged_fraction": 0.0,
        },
    )
    monkeypatch.setattr(
        gx, "state_bounds_utm", lambda ee_, geom: (700_123.4, 2_400_050.9, 760_999.9, 2_470_321.0)
    )
    monkeypatch.setattr(gx, "build_composite", lambda ee_, col, geom: MagicMock())
    return ee


def test_dry_run_never_starts_task_and_warns(monkeypatch, capsys):
    ee = _patch_pipeline(monkeypatch)
    assert gx.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Scenes: 12" in out and "WARNING" in out and "Dry run" in out
    assert "dummy-test-project" not in out
    ee.batch.Export.image.toDrive.assert_not_called()


def test_export_passes_snapped_transform_and_starts_task(monkeypatch, capsys):
    ee = _patch_pipeline(monkeypatch)
    assert gx.main([]) == 0
    kwargs = ee.batch.Export.image.toDrive.call_args.kwargs
    assert kwargs["crs"] == "EPSG:32613" and kwargs["folder"] == "ags-hydroscope"
    assert kwargs["fileNamePrefix"] == "s2_ags_2024_04"
    assert kwargs["crsTransform"][2] % 640 == 0 and kwargs["crsTransform"][5] % 640 == 0
    assert "scale" not in kwargs
    ee.batch.Export.image.toDrive.return_value.start.assert_called_once()
    out = capsys.readouterr().out
    assert "Task started" in out and "earthengine task list" in out
    assert "dummy-test-project" not in out
