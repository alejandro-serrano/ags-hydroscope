"""Raster check script: statistics, report rendering and an end-to-end run on a tiny raster."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hydroscope.data.eurosat import BAND_SETS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_ags_raster", ROOT / "scripts/check_ags_raster.py"
)
car = importlib.util.module_from_spec(spec)
spec.loader.exec_module(car)

X0, Y0 = 640 * 800, 640 * 3750  # on the 640 m lattice
needs_gdal = pytest.mark.skipif(car.find_gdalbuildvrt() is None, reason="gdalbuildvrt not found")


# ---------------------------------------------------------------------------------------------
# Statistics on synthetic arrays
# ---------------------------------------------------------------------------------------------


def test_valid_mask():
    vo = np.array([[0, 1], [3, 0]], dtype=np.uint16)
    assert car.valid_mask(vo).tolist() == [[False, True], [True, False]]


def test_band_percentiles_ignore_invalid_pixels():
    band = np.array([0, 0, 100, 200, 300, 400, 500, 65535], dtype=np.uint16)
    valid = np.array([0, 0, 1, 1, 1, 1, 1, 0], dtype=bool)
    p2, med, p98 = car.band_percentiles(band, valid)
    assert med == 300.0
    assert 100.0 < p2 < 120.0 and 480.0 < p98 < 500.0


def test_band_percentiles_no_valid_pixels_is_nan():
    out = car.band_percentiles(np.zeros(4, np.uint16), np.zeros(4, bool))
    assert len(out) == 3 and all(np.isnan(v) for v in out)


def test_zscore_and_flag():
    z = car.zscore(1500.0, 1000.0, 200.0)
    assert z == pytest.approx(2.5)
    assert car.flag_z(z) and car.flag_z(-2.5)
    assert not car.flag_z(2.0) and not car.flag_z(float("nan"))
    assert np.isnan(car.zscore(1.0, 0.0, 0.0))


def test_fraction_within():
    band = np.array([1000, 1100, 1300, 1700, 5000], dtype=np.uint16)
    valid = np.array([1, 1, 1, 1, 0], dtype=bool)
    # mean 1000, std 100 -> within 300: 1000, 1100, 1300 but not 1700; invalid ignored
    assert car.fraction_within(band, valid, 1000.0, 100.0) == pytest.approx(0.75)
    assert np.isnan(car.fraction_within(band, np.zeros(5, bool), 1000.0, 100.0))


def test_summarize_valid_obs_proxy_mode():
    vo = np.array([[0, 0, 2], [4, 6, 0]], dtype=np.uint16)
    s = car.summarize_valid_obs(vo)
    assert s["distinguishable"] is False
    assert (s["min"], s["max"], s["n_pixels"], s["n_zero"]) == (0, 6, 6, 3)
    assert s["fraction_zero"] == pytest.approx(0.5)


def test_summarize_valid_obs_with_state_mask():
    vo = np.array([[0, 0, 2], [4, 6, 0]], dtype=np.uint16)
    inside = np.array([[0, 1, 1], [1, 1, 0]], dtype=bool)
    s = car.summarize_valid_obs(vo, inside)
    assert s["distinguishable"] is True
    assert (s["min"], s["median"], s["max"]) == (0, 3.0, 6)
    assert (s["n_pixels"], s["n_zero"]) == (4, 1)
    assert s["fraction_zero"] == pytest.approx(0.25)


def test_summarize_valid_obs_all_zero():
    s = car.summarize_valid_obs(np.zeros((3, 3), np.uint16))
    assert s["fraction_zero"] == 1.0 and s["max"] == 0


def test_summarize_valid_pixels_distribution():
    vo = np.array([[0, 0, 4], [5, 5, 6]], dtype=np.uint16)
    s = car.summarize_valid_pixels(vo)
    assert (s["min"], s["median"], s["max"]) == (4, 5.0, 6)
    assert s["counts"] == {4: 1, 5: 2, 6: 1}
    assert s["area_km2"] == pytest.approx(4 * 100 / 1e6)
    assert car.summarize_valid_pixels(np.zeros((2, 2), np.uint16))["counts"] == {}


@pytest.mark.parametrize(
    ("left", "top", "expected"),
    [
        (640 * 800, 640 * 3750, (0, 0)),  # on the lattice
        (719_690, 2_486_210, (31, 45)),  # real export: cropped to the state, 10 m aligned
        (719_695, 2_486_210, None),  # not a multiple of the pixel size
    ],
)
def test_grid_offset_px(left, top, expected):
    assert car.grid_offset_px(left, top) == expected


def test_stretch_maps_percentiles_to_unit_interval():
    band = np.arange(101, dtype=np.float32)
    out = car.stretch(band, np.ones(101, bool))
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[50] == pytest.approx(0.5, abs=0.01)
    assert not car.stretch(band, np.zeros(101, bool)).any()


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def _rows():
    ok = car.compare_band(
        np.full(10, 1000, np.uint16), np.ones(10, bool), mean=1000.0, std=100.0, name="B01"
    )
    bad = car.compare_band(
        np.full(10, 3000, np.uint16), np.ones(10, bool), mean=1000.0, std=100.0, name="B02"
    )
    return [ok, bad]


def _info(aligned=True):
    return {
        "source": "data/ags/raw/s2_ags_2024_04.vrt",
        "n_tiles": 2,
        "width": 100,
        "height": 80,
        "crs": "EPSG:32613",
        "pixel_size": 10,
        "dtype": "uint16",
        "bands": 14,
        "origin_aligned": aligned,
        "grid_offset_px": (0, 0) if aligned else None,
    }


def test_render_report_contents():
    rows = _rows()
    obs = car.summarize_valid_obs(np.array([[0, 2], [3, 4]], np.uint16))
    obs.update(n_valid=3, n_total=4)
    obs["valid_pixels"] = car.summarize_valid_pixels(np.array([[0, 2], [3, 4]], np.uint16))
    warnings = car.collect_warnings(rows, obs, _info())
    text = car.render_report(_info(), rows, obs, warnings, "figures/ags_2024_04_rgb.png")
    assert text.startswith(
        "<!-- Generated by scripts/check_ags_raster.py. Do not edit by hand. -->"
    )
    assert "| B01 |" in text and "| B02 |" in text
    assert "FLAG" in text and "Flagged bands (|z| > 2): B02." in text
    assert "not distinguishable" in text
    assert "25.00 %" in text  # fraction of valid_obs == 0
    assert "![RGB preview" in text
    assert "First full 640 m cell at pixel offset: col 0, row 0" in text
    assert "| 2 | 1 | 33.33 % |" in text  # valid_obs distribution over valid pixels
    assert any("B02" in w for w in warnings)


def test_collect_warnings_origin_misaligned_and_clean_case():
    obs = {"fraction_zero": 0.0, "distinguishable": True}
    assert car.collect_warnings([], obs, _info()) == []
    assert any("640" in w for w in car.collect_warnings([], obs, _info(aligned=False)))


# ---------------------------------------------------------------------------------------------
# Raster properties and end-to-end on tiny GeoTIFFs
# ---------------------------------------------------------------------------------------------


def _write_tile(path, x0, count=14, dtype="uint16", crs="EPSG:32613", res=10, seed=0, size=64):
    """Tiny tile with EuroSAT-like values; band 14 (valid_obs) has a zero strip in the corner."""
    rng = np.random.default_rng(seed)
    data = np.zeros((count, size, size), dtype=dtype)
    for i in range(min(count, 13)):
        data[i] = rng.integers(500 + 100 * i, 1500 + 100 * i, (size, size))
    if count == 14:
        data[13] = rng.integers(1, 5, (size, size))
        data[13, :, :4] = 0
        data[:13, :, :4] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=from_origin(x0, Y0, res, res),
    ) as dst:
        dst.write(data)
    return path


def test_check_raster_properties_ok(tmp_path):
    path = _write_tile(tmp_path / "t.tif", X0)
    with rasterio.open(path) as src:
        info = car.check_raster_properties(src)
    assert info["bands"] == 14 and info["origin_aligned"] is True
    assert (info["width"], info["height"]) == (64, 64)


def test_check_raster_properties_cropped_origin_is_aligned(tmp_path):
    # Earth Engine crops the export to the region: the origin leaves the 640 m lattice but stays
    # on the 10 m pixel lattice, so grid cells still map to whole pixel windows.
    path = _write_tile(tmp_path / "t.tif", X0 + 330)
    info = car.check_vrt_properties(path)
    assert info["origin_aligned"] is True and info["grid_offset_px"] == (31, 0)


def test_check_raster_properties_origin_misaligned(tmp_path):
    path = _write_tile(tmp_path / "t.tif", X0 + 5)
    info = car.check_vrt_properties(path)
    assert info["origin_aligned"] is False and info["grid_offset_px"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 13}, "14 bands"),
        ({"dtype": "int16"}, "uint16"),
        ({"crs": "EPSG:32614"}, "EPSG:32613"),
        ({"res": 20}, "10 m"),
    ],
)
def test_check_raster_properties_rejects(tmp_path, kwargs, message):
    path = _write_tile(tmp_path / "bad.tif", X0, **kwargs)
    with pytest.raises(ValueError, match=message):
        car.check_vrt_properties(path)


def _write_stats(path):
    stats = {b: {"mean": 1000.0 + 100 * i, "std": 300.0} for i, b in enumerate(BAND_SETS["ms13"])}
    path.write_text(json.dumps(stats))
    return path


def test_load_eurosat_stats_missing_file_has_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_eurosat"):
        car.load_eurosat_stats(tmp_path / "nope.json")


def test_main_stops_when_stats_missing(tmp_path, capsys):
    rc = car.main(
        [
            "--raw-dir",
            str(tmp_path),
            "--stats",
            str(tmp_path / "nope.json"),
            "--report",
            str(tmp_path / "r.md"),
            "--preview",
            str(tmp_path / "p.png"),
        ]
    )
    assert rc == 1
    assert "prepare_eurosat" in capsys.readouterr().err
    assert not (tmp_path / "r.md").exists()


def test_run_check_without_tiles(tmp_path):
    stats = _write_stats(tmp_path / "stats.json")
    with pytest.raises(FileNotFoundError, match="s2_ags_2024_04"):
        car.run_check(tmp_path, stats, tmp_path / "r.md", tmp_path / "p.png")


@needs_gdal
def test_build_vrt_mosaics_adjacent_tiles(tmp_path):
    a = _write_tile(tmp_path / "s2_ags_2024_04-0000000000-0000000000.tif", X0)
    b = _write_tile(tmp_path / "s2_ags_2024_04-0000000000-0000000064.tif", X0 + 640, seed=1)
    vrt = car.build_vrt([a, b], tmp_path / "out.vrt")
    info = car.check_vrt_properties(vrt)
    assert (info["width"], info["height"], info["bands"]) == (128, 64, 14)


@needs_gdal
@pytest.mark.parametrize("with_boundary", [False, True])
def test_run_check_end_to_end(tmp_path, with_boundary):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_tile(raw / "s2_ags_2024_04-0000000000-0000000000.tif", X0)
    _write_tile(raw / "s2_ags_2024_04-0000000000-0000000064.tif", X0 + 640, seed=1)
    stats = _write_stats(tmp_path / "stats.json")
    boundary = None
    if with_boundary:
        import geopandas as gpd
        from shapely.geometry import box

        # in-state = left tile only
        boundary = tmp_path / "state.gpkg"
        gdf = gpd.GeoDataFrame(geometry=[box(X0, Y0 - 640, X0 + 640, Y0)], crs="EPSG:32613")
        gdf.to_file(boundary, driver="GPKG")

    report, png = tmp_path / "docs" / "check.md", tmp_path / "docs" / "figures" / "rgb.png"
    result = car.run_check(raw, stats, report, png, boundary=boundary)

    assert result["info"]["width"] == 128 and result["info"]["n_tiles"] == 2
    assert [r["band"] for r in result["rows"]] == list(BAND_SETS["ms13"])
    assert all(np.isfinite(r["median"]) for r in result["rows"])
    obs = result["obs"]
    assert obs["n_total"] == 128 * 64
    assert obs["n_valid"] == 2 * 64 * 60  # 4 empty columns per tile
    if with_boundary:
        assert obs["distinguishable"] and obs["n_pixels"] == 64 * 64
        assert obs["n_zero"] == 64 * 4
    else:
        assert not obs["distinguishable"] and obs["n_zero"] == 2 * 64 * 4

    text = report.read_text()
    assert text.startswith(car.HEADER)
    assert "figures/rgb.png" in text and png.stat().st_size > 0
    assert (raw / "s2_ags_2024_04.vrt").exists()
