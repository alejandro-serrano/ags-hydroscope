"""Tiling: windows, filters, fixed stretch, patch contract and end-to-end run on a tiny raster."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from hydroscope.geo import grid as g
from hydroscope.geo import tiling as t
from hydroscope.geo.gee_export import boundary_frame, eurosat_band_names

LX, LY = 640 * 800, 640 * 3750
WIDTH, HEIGHT = 320, 256  # 5 x 4 cells of 64 px
TRANSFORM = from_origin(LX + 330, LY + 450, 10, 10)
STATE = box(LX + 640, LY - 1920, LX + 3200, LY)  # 4 x 3 cells


def make_data() -> np.ndarray:
    """14-band array: band b = 1000 + 50 b + column parity; valid_obs = 2 with a hole."""
    data = np.zeros((14, HEIGHT, WIDTH), dtype=np.uint16)
    cols = np.arange(WIDTH)[None, :] % 7
    rows = np.arange(HEIGHT)[:, None] % 5
    for b in range(13):
        data[b] = 1000 + 50 * b + cols * 10 + rows
    data[13] = 2
    return data


def write_raster(path, data=None):
    data = make_data() if data is None else data
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=14,
        dtype="uint16",
        crs="EPSG:32613",
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def grid_file(tmp_path):
    state = gpd.GeoDataFrame({"name": ["x"]}, geometry=[STATE], crs="EPSG:32613")
    grid = g.build_grid(state, TRANSFORM)
    path = tmp_path / "grid.gpkg"
    grid.to_file(path, driver="GPKG", layer="grid")
    return path


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def test_cell_window_is_64_by_64():
    w = t.cell_window(31, 45)
    assert (w.col_off, w.row_off, w.width, w.height) == (31, 45, 64, 64)


def test_valid_fraction():
    vo = np.zeros((4, 4), np.uint16)
    vo[:2] = 3
    assert t.valid_fraction(vo) == 0.5
    assert t.valid_fraction(np.zeros((2, 2))) == 0.0


@pytest.mark.parametrize(
    ("frac", "valid", "expected"),
    [(1.0, 1.0, True), (0.8, 0.6, True), (0.79, 1.0, False), (1.0, 0.59, False)],
)
def test_passes_filters(frac, valid, expected):
    assert t.passes_filters(frac, valid) is expected


def test_extract_window_zero_pads_outside_strip():
    strip = np.ones((2, 3, 100), dtype=np.uint16)
    out = t.extract_window(strip, 80)
    assert out.shape == (2, 3, 64)
    assert out[:, :, :20].all() and not out[:, :, 20:].any()
    left = t.extract_window(strip, -10)
    assert not left[:, :, :10].any() and left[:, :, 10:].all()


def test_fixed_stretch_same_dn_same_colour_in_two_patches():
    limits = {"B04": (1000.0, 3000.0), "B03": (800.0, 2000.0), "B02": (500.0, 1500.0)}
    a = np.full((3, 64, 64), 2000, dtype=np.uint16)
    b = a.copy()
    b[:, :10] = 100  # a very different patch must not change the colour of the rest
    pa, pb = t.render_preview(a, limits), t.render_preview(b, limits)
    assert (pa[200, 200] == pb[200, 200]).all()
    assert pa[200, 200, 0] == round(255 * (2000 - 1000) / 2000)


def test_preview_shape_dtype_and_nearest_upscale():
    limits = {"B04": (0.0, 100.0), "B03": (0.0, 100.0), "B02": (0.0, 100.0)}
    rgb = np.zeros((3, 64, 64), dtype=np.uint16)
    rgb[:, 0, 0] = 100
    img = t.render_preview(rgb, limits)
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    assert (img[:4, :4] == 255).all() and (img[4:, 4:] == 0).all()


def test_apply_stretch_clips_and_handles_degenerate_limits():
    out = t.apply_stretch(np.array([0, 50, 200]), 0, 100)
    assert out.tolist() == [0, 128, 255]
    assert not t.apply_stretch(np.array([1, 2]), 5, 5).any()


def test_boundary_frame_from_geojson_dict():
    geojson = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    frame = boundary_frame(geojson)
    assert frame.crs.to_epsg() == 4326 and len(frame) == 1
    assert frame["name"].iloc[0] == "Aguascalientes"
    assert frame.geometry.iloc[0].area == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------


def test_rgb_stretch_limits_use_only_valid_pixels(tmp_path):
    data = make_data()
    data[13, :, :100] = 0  # invalid pixels carry extreme values that must be ignored
    data[3, :, :100] = 60000
    path = write_raster(tmp_path / "r.tif", data)
    with rasterio.open(path) as src:
        limits = t.rgb_stretch_limits(src)
    assert set(limits) == {"B04", "B03", "B02"}
    lo, hi = limits["B04"]
    assert hi < 3000 and lo >= 1000 + 150


def test_write_patch_contract(tmp_path):
    data = np.arange(13 * 64 * 64, dtype=np.uint16).reshape(13, 64, 64)
    path = tmp_path / "p.tif"
    t.write_patch(path, data, from_origin(1000, 5000, 10, 10))
    with rasterio.open(path) as src:
        assert (src.count, src.height, src.width) == (13, 64, 64)
        assert set(src.dtypes) == {"uint16"} and src.crs.to_epsg() == 32613
        assert list(src.descriptions) == eurosat_band_names()
        assert src.compression.name.lower() == "deflate"
        np.testing.assert_array_equal(src.read(), data)
    with pytest.raises(ValueError):
        t.write_patch(path, data.astype(np.float32), from_origin(0, 0, 10, 10))


def test_tile_end_to_end(tmp_path, grid_file):
    vrt = write_raster(tmp_path / "raw.tif")
    out = tmp_path / "ags"
    cells = t.tile(vrt, grid_file, out)

    assert list(cells.columns) == t.CELLS_COLUMNS
    assert len(cells) == 12 and cells["cell_id"].is_unique
    assert cells["has_patch"].all() and cells["valid_frac"].eq(1.0).all()
    assert cells.groupby("cell_id")["block_id"].nunique().eq(1).all()
    csv = pd.read_csv(out / "cells.csv")
    assert list(csv.columns) == t.CELLS_COLUMNS and len(csv) == 12
    assert csv["lat"].between(20, 24).all() and csv["lon"].between(-106, -104).all()

    stretch = json.loads((out / "previews" / "stretch_rgb.json").read_text())
    assert set(stretch) == {"B04", "B03", "B02"}

    # Patch of cell r0001_c0002: bounds equal the cell geometry; data equal the raster window.
    cell = gpd.read_file(grid_file, layer="grid").set_index("cell_id").loc["r0001_c0002"]
    with rasterio.open(out / "patches" / "r0001_c0002.tif") as src:
        assert src.shape == (64, 64) and src.count == 13 and src.crs.to_epsg() == 32613
        assert src.dtypes[0] == "uint16"
        assert tuple(src.bounds) == pytest.approx(cell.geometry.bounds)
        expected = make_data()[:13, 45 + 64 : 45 + 128, 31 + 128 : 31 + 192]
        np.testing.assert_array_equal(src.read(), expected)

    png = out / "previews" / "r0001_c0002.png"
    from PIL import Image

    with Image.open(png) as im:
        assert im.mode == "RGB"
        img = np.asarray(im)
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8


def test_tile_filters_cells_with_few_valid_pixels(tmp_path, grid_file):
    data = make_data()
    # Cell r0000_c0000 spans rows 45..108, cols 31..94: invalidate 50 % of it.
    data[13, 45:77, 31:95] = 0
    data[:13, 45:77, 31:95] = 0
    vrt = write_raster(tmp_path / "raw.tif", data)
    cells = t.tile(vrt, grid_file, tmp_path / "ags").set_index("cell_id")
    assert cells.loc["r0000_c0000", "valid_frac"] == pytest.approx(0.5)
    assert not cells.loc["r0000_c0000", "has_patch"]
    assert not (tmp_path / "ags" / "patches" / "r0000_c0000.tif").exists()
    assert not (tmp_path / "ags" / "previews" / "r0000_c0000.png").exists()
    assert cells["has_patch"].sum() == 11


def test_tile_skips_cells_mostly_outside_state(tmp_path):
    half = gpd.GeoDataFrame(
        {"name": ["x"]},
        geometry=[box(LX + 640, LY - 1920, LX + 3200 - 320, LY)],  # last column half covered
        crs="EPSG:32613",
    )
    path = tmp_path / "grid.gpkg"
    g.build_grid(half, TRANSFORM).to_file(path, driver="GPKG", layer="grid")
    cells = t.tile(write_raster(tmp_path / "raw.tif"), path, tmp_path / "ags")
    last = cells[cells["cell_id"].str.endswith("c0003")]
    assert last["frac_in_state"].eq(0.5).all() and not last["has_patch"].any()
    assert cells["has_patch"].sum() == 9


def test_tile_limit(tmp_path, grid_file):
    cells = t.tile(write_raster(tmp_path / "raw.tif"), grid_file, tmp_path / "ags", limit=3)
    assert len(cells) == 3
