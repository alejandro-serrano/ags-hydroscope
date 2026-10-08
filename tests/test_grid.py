"""640 m grid: ids, blocks, alignment with the raster pixel lattice and state coverage."""

import geopandas as gpd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from hydroscope.geo import grid as g

LX, LY = 640 * 800, 640 * 3750  # a point on the 640 m lattice
# Raster origin like the real export: multiple of 10 m, not of 640 m (offset 31 x 45 px).
RASTER = from_origin(LX + 330, LY + 450, 10, 10)


def state_frame(geom=None) -> gpd.GeoDataFrame:
    """State = 4 x 3 cells whose west edge is 640 m east of the lattice point."""
    geom = geom if geom is not None else box(LX + 640, LY - 1920, LX + 3200, LY)
    return gpd.GeoDataFrame({"name": ["x"]}, geometry=[geom], crs="EPSG:32613")


def test_id_formats():
    assert g.cell_id(3, 12) == "r0003_c0012"
    assert g.block_id(3, 12) == "b000_001"
    assert g.block_id(1234, 56) == "b154_007"


def test_block_boundary_between_row_7_and_8():
    assert g.block_id(7, 8) == "b000_001"
    assert g.block_id(8, 8) == "b001_001"
    assert g.block_id(7, 7) == "b000_000"


def test_pixel_offset_of_real_like_origin():
    assert g.pixel_offset(LX + 640, LY, RASTER) == (31, 45)


def test_grid_origin_is_multiple_of_pixel_from_raster_origin():
    grid = g.build_grid(state_frame(), RASTER)
    x0, y0 = grid.attrs["x0"], grid.attrs["y0"]
    assert (x0, y0) == (LX + 640, LY)
    assert (x0 - RASTER.c) % 10 == 0 and (RASTER.f - y0) % 10 == 0
    assert g.pixel_offset(x0, y0, RASTER) == (31, 45)


def test_pixel_offset_rejects_5_m_shift():
    shifted = from_origin(LX + 335, LY + 450, 10, 10)
    with pytest.raises(ValueError, match="pixel lattice"):
        g.pixel_offset(LX + 640, LY, shifted)
    with pytest.raises(ValueError):
        g.build_grid(state_frame(), shifted)


def test_build_grid_covers_state_with_expected_cells():
    grid = g.build_grid(state_frame(), RASTER)
    assert list(grid.columns) == g.GRID_COLUMNS
    assert len(grid) == 12
    assert grid.crs.to_epsg() == 32613
    assert grid["frac_in_state"].eq(1.0).all()
    first = grid.iloc[0]
    assert first["cell_id"] == "r0000_c0000" and first["row"] == 0 and first["col"] == 0
    assert first.geometry.bounds == (LX + 640, LY - 640, LX + 1280, LY)  # row 0 = north
    assert all(b.area == 640 * 640 for b in grid.geometry)


def test_frac_in_state_half_covered_cell():
    state = state_frame(box(LX + 640, LY - 640, LX + 960, LY))  # west half of one cell
    grid = g.build_grid(state, RASTER)
    assert len(grid) == 1
    assert grid["frac_in_state"].iloc[0] == pytest.approx(0.5)


def test_cells_only_touching_the_state_are_dropped():
    grid = g.build_grid(state_frame(), RASTER)
    assert grid["row"].max() == 2 and grid["col"].max() == 3


def test_state_in_other_crs_is_reprojected():
    grid = g.build_grid(state_frame().to_crs("EPSG:4326"), RASTER)
    assert len(grid) >= 12  # round trip may spill a few millimetres into neighbours


def test_cell_ids_unique_and_never_in_two_blocks():
    # Large state so that several 8 x 8 blocks are involved.
    state = state_frame(box(LX, LY - 640 * 20, LX + 640 * 20, LY))
    grid = g.build_grid(state, None)
    assert grid["cell_id"].is_unique
    assert grid.groupby("cell_id")["block_id"].nunique().eq(1).all()
    assert grid["block_id"].nunique() == 9  # 20 / 8 -> 3 x 3 blocks
    sizes = grid.groupby("block_id").size()
    assert sizes.max() == 64
