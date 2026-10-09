"""Fixed 640 m grid of Aguascalientes in EPSG:32613 (contract 5 in ``docs/contracts.md``).

Usage::

    python -m hydroscope.geo.grid --boundary data/ref/ags_state.gpkg \\
        [--vrt data/ags/raw/s2_ags_2024_04.vrt] [--out data/ref/grid.gpkg]

The grid is anchored on the **absolute** 640 m lattice of EPSG:32613 (origin = state bounds
snapped outward with :func:`hydroscope.geo.gee_export.snap_extent`), so ``cell_id`` values stay
stable across the monthly exports of stage 2. The raster origin only has to be a multiple of the
10 m pixel size; :func:`pixel_offset` raises ``ValueError`` otherwise, which guarantees that every
cell is exactly one 64 x 64 pixel window.

Ids: ``cell_id = r{row:04d}_c{col:04d}`` (row 0 = north, col 0 = west) and
``block_id = b{row//8:03d}_{col//8:03d}`` (blocks of 8 x 8 cells = 5.12 km).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from affine import Affine
from shapely.geometry import box

from hydroscope.geo.gee_export import EXPORT_CRS, GRID_M, PIXEL_SIZE_M, snap_extent

BLOCK_CELLS = 8
"""Cells per block side: blocks are 8 x 8 cells = 5.12 km."""
GRID_LAYER = "grid"
GRID_COLUMNS = ["cell_id", "row", "col", "block_id", "frac_in_state", "geometry"]
CELL_PIXELS = GRID_M // PIXEL_SIZE_M
"""Pixels per cell side (64)."""


def cell_id(row: int, col: int) -> str:
    """Cell id ``r{row:04d}_c{col:04d}`` (row 0 = north, col 0 = west)."""
    return f"r{row:04d}_c{col:04d}"


def block_id(row: int, col: int, block_cells: int = BLOCK_CELLS) -> str:
    """Block id ``b{row//8:03d}_{col//8:03d}`` of the 8 x 8-cell block containing the cell."""
    return f"b{row // block_cells:03d}_{col // block_cells:03d}"


def grid_origin(bounds: Sequence[float]) -> tuple[int, int]:
    """North-west corner ``(x0, y0)`` of the grid: ``bounds`` snapped outward to the 640 m lattice.

    Args:
        bounds: ``(x_min, y_min, x_max, y_max)`` of the state in EPSG:32613 metres.
    """
    x0, y0, _, _ = snap_extent(bounds)
    return x0, y0


def pixel_offset(x0: float, y0: float, transform: Affine) -> tuple[int, int]:
    """Pixel offset ``(col_off, row_off)`` of the point ``(x0, y0)`` from a raster transform.

    Args:
        x0: easting of a north-west corner (m).
        y0: northing of a north-west corner (m).
        transform: north-up affine transform of the raster (10 m pixels).

    Raises:
        ValueError: if either offset is not an exact multiple of the 10 m pixel, in which case
            grid cells would not map to whole pixel windows.
    """
    col = (x0 - transform.c) / abs(transform.a)
    row = (transform.f - y0) / abs(transform.e)
    if not (np.isclose(col, round(col), atol=1e-6) and np.isclose(row, round(row), atol=1e-6)):
        raise ValueError(
            f"point ({x0}, {y0}) is not on the pixel lattice of the raster (origin "
            f"({transform.c}, {transform.f}), pixel {abs(transform.a)} m): offsets "
            f"({col:.3f}, {row:.3f}) px are not integers"
        )
    return int(round(col)), int(round(row))


def cell_box(x0: float, y0: float, row: int, col: int, size: float = GRID_M) -> Any:
    """Shapely box of the ``size`` m cell ``(row, col)`` of a grid with north-west corner x0, y0."""
    west = x0 + col * size
    north = y0 - row * size
    return box(west, north - size, west + size, north)


def cell_outline_latlon(geom_utm: Any) -> list[tuple[float, float]]:
    """Closed outline of a cell as ``(lat, lon)`` pairs in EPSG:4326 (5 points, for ipyleaflet).

    Args:
        geom_utm: cell polygon in EPSG:32613.
    """
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs(EXPORT_CRS, "EPSG:4326", always_xy=True)
    xs, ys = geom_utm.exterior.coords.xy
    lons, lats = to_wgs84.transform(list(xs), list(ys))
    return [(float(lat), float(lon)) for lat, lon in zip(lats, lons, strict=True)]


def build_grid(state: gpd.GeoDataFrame, raster_transform: Affine | None = None) -> gpd.GeoDataFrame:
    """Build the 640 m grid over the state.

    Args:
        state: state geometry in any CRS (reprojected to EPSG:32613).
        raster_transform: transform of the export raster; when given, the grid origin must lie on
            its 10 m pixel lattice (see :func:`pixel_offset`).

    Returns:
        ``GeoDataFrame`` in EPSG:32613 with columns ``cell_id, row, col, block_id, frac_in_state,
        geometry`` for every cell intersecting the state, sorted by (row, col). The grid origin is
        stored in ``attrs["x0"]`` and ``attrs["y0"]``.

    Raises:
        ValueError: if the grid origin is not on the raster's pixel lattice or the state is empty.
    """
    union = state.to_crs(EXPORT_CRS).geometry.union_all()
    if union.is_empty:
        raise ValueError("empty state geometry")
    x0, y0, x1, y1 = snap_extent(union.bounds)
    if raster_transform is not None:
        pixel_offset(x0, y0, raster_transform)
    n_cols = (x1 - x0) // GRID_M
    n_rows = (y0 - y1) // GRID_M

    rows, cols = np.divmod(np.arange(n_rows * n_cols), n_cols)
    west = x0 + cols * GRID_M
    north = y0 - rows * GRID_M
    cells = shapely.box(west, north - GRID_M, west + GRID_M, north)
    shapely.prepare(union)
    keep = shapely.intersects(cells, union)
    rows, cols, cells = rows[keep], cols[keep], cells[keep]
    frac = shapely.area(shapely.intersection(cells, union)) / float(GRID_M**2)
    keep = frac > 0  # drop cells that only touch the boundary
    rows, cols, cells, frac = rows[keep], cols[keep], cells[keep], frac[keep]

    frame = gpd.GeoDataFrame(
        {
            "cell_id": [cell_id(int(r), int(c)) for r, c in zip(rows, cols, strict=True)],
            "row": rows.astype(int),
            "col": cols.astype(int),
            "block_id": [block_id(int(r), int(c)) for r, c in zip(rows, cols, strict=True)],
            "frac_in_state": frac,
        },
        geometry=cells,
        crs=EXPORT_CRS,
    )
    frame.attrs.update(x0=x0, y0=y0)
    return frame[GRID_COLUMNS]


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m hydroscope.geo.grid", description="Build the 640 m grid of Aguascalientes."
    )
    parser.add_argument("--boundary", type=Path, required=True, help="state GPKG (layer 'state')")
    parser.add_argument(
        "--vrt",
        type=Path,
        default=Path("data/ags/raw/s2_ags_2024_04.vrt"),
        help="export raster, opened only to validate the alignment",
    )
    parser.add_argument("--out", type=Path, default=Path("data/ref/grid.gpkg"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    import rasterio

    args = build_parser().parse_args(argv)
    if not args.boundary.exists():
        print(f"ERROR: boundary file not found: {args.boundary}", file=sys.stderr)
        return 1
    state = gpd.read_file(args.boundary)
    transform = None
    if args.vrt.exists():
        with rasterio.open(args.vrt) as src:
            transform = src.transform
    else:
        print(f"WARNING: {args.vrt} not found; pixel alignment not validated")
    try:
        grid = build_grid(state, transform)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    x0, y0 = grid.attrs["x0"], grid.attrs["y0"]
    print(f"Grid origin {EXPORT_CRS}: x0={x0}, y0={y0}")
    if transform is not None:
        print(
            "Pixel offset of the grid origin in the raster (col, row): "
            f"{pixel_offset(x0, y0, transform)}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(args.out, driver="GPKG", layer=GRID_LAYER)
    print(f"{len(grid)} cells in {grid['block_id'].nunique()} blocks -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
