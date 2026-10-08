"""Tile the Earth Engine export into one 13 x 64 x 64 patch per 640 m grid cell.

Usage::

    python -m hydroscope.geo.tiling [--vrt data/ags/raw/s2_ags_2024_04.vrt] \\
        [--grid data/ref/grid.gpkg] [--out-dir data/ags] [--limit N]

Outputs (contracts 1 and 6 in ``docs/contracts.md``):

* ``{out_dir}/patches/{cell_id}.tif``  13-band uint16 GeoTIFF (B01 ... B12, B8A), deflate;
* ``{out_dir}/previews/{cell_id}.png`` 256 x 256 RGB preview (B04, B03, B02);
* ``{out_dir}/previews/stretch_rgb.json`` the global 2-98 % stretch limits of the preview;
* ``{out_dir}/cells.csv`` one row per grid cell.

A patch and its preview are written only if ``frac_in_state >= 0.8`` and ``valid_frac >= 0.6``
(``valid_frac`` = fraction of pixels with ``valid_obs >= 1``). The preview stretch is computed once
over all valid pixels of the whole raster, never per patch, so equal DN give equal colour in every
preview. The raster is read in strips of 64 rows (one strip per grid row) instead of per cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.windows import Window

from hydroscope.geo.gee_export import EXPORT_CRS, PIXEL_SIZE_M, eurosat_band_names
from hydroscope.geo.grid import CELL_PIXELS, pixel_offset

SPECTRAL_BANDS: list[str] = eurosat_band_names()
N_SPECTRAL = len(SPECTRAL_BANDS)
VALID_BAND_INDEX = 14
"""1-based index of the ``valid_obs`` band in the export."""
RGB: tuple[str, str, str] = ("B04", "B03", "B02")
MIN_FRAC_IN_STATE = 0.8
MIN_VALID_FRAC = 0.6
PREVIEW_SIZE = 256
STRETCH_PERCENTILES = (2.0, 98.0)
CELLS_COLUMNS = ["cell_id", "block_id", "frac_in_state", "valid_frac", "lat", "lon", "has_patch"]


# --------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------


def cell_window(col_off: int, row_off: int) -> Window:
    """64 x 64 window of the cell whose north-west corner is at pixel ``(col_off, row_off)``."""
    return Window(col_off, row_off, CELL_PIXELS, CELL_PIXELS)


def valid_fraction(valid_obs: np.ndarray) -> float:
    """Fraction of pixels with at least one cloud-free observation (``valid_obs >= 1``)."""
    return float(np.mean(np.asarray(valid_obs) >= 1))


def passes_filters(
    frac_in_state: float,
    valid_frac: float,
    min_in_state: float = MIN_FRAC_IN_STATE,
    min_valid: float = MIN_VALID_FRAC,
) -> bool:
    """True if a cell gets a patch: enough area in the state and enough valid pixels."""
    return frac_in_state >= min_in_state and valid_frac >= min_valid


def extract_window(strip: np.ndarray, col_off: int, size: int = CELL_PIXELS) -> np.ndarray:
    """Columns ``[col_off, col_off + size)`` of a ``(bands, rows, width)`` strip, zero-padded.

    Columns outside the strip are filled with 0, so ``valid_obs == 0`` marks them.
    """
    bands, rows, width = strip.shape
    out = np.zeros((bands, rows, size), dtype=strip.dtype)
    lo, hi = max(col_off, 0), min(col_off + size, width)
    if hi > lo:
        out[:, :, lo - col_off : hi - col_off] = strip[:, :, lo:hi]
    return out


def apply_stretch(band: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear stretch of ``band`` to uint8 with fixed limits ``lo`` and ``hi`` (clipped)."""
    if hi <= lo:
        return np.zeros(np.shape(band), dtype=np.uint8)
    scaled = np.clip((np.asarray(band, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255).astype(np.uint8)


def render_preview(
    rgb: np.ndarray,
    limits: Mapping[str, tuple[float, float]],
    size: int = PREVIEW_SIZE,
) -> np.ndarray:
    """RGB preview ``(size, size, 3)`` uint8 from ``(3, H, W)`` bands in order B04, B03, B02.

    Uses the fixed, global ``limits`` and upscales with nearest neighbour (``np.repeat``).
    """
    channels = [apply_stretch(rgb[i], *limits[name]) for i, name in enumerate(RGB)]
    img = np.stack(channels, axis=-1)
    factor = size // img.shape[0]
    if factor < 1 or factor * img.shape[0] != size or factor * img.shape[1] != size:
        raise ValueError(
            f"patch of {img.shape[:2]} px cannot be upscaled to {size} px by an integer"
        )
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


# --------------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------------


def read_strip(src: Any, row_off: int, height: int = CELL_PIXELS) -> np.ndarray:
    """Read all bands of ``height`` rows starting at ``row_off`` over the full width.

    Rows outside the raster are zero-filled.
    """
    out = np.zeros((src.count, height, src.width), dtype=np.uint16)
    lo, hi = max(row_off, 0), min(row_off + height, src.height)
    if hi > lo:
        out[:, lo - row_off : hi - row_off, :] = src.read(window=Window(0, lo, src.width, hi - lo))
    return out


def rgb_stretch_limits(src: Any) -> dict[str, tuple[float, float]]:
    """Global 2-98 % limits of B04, B03 and B02 over all pixels with ``valid_obs >= 1``.

    Bands are read one at a time. The limits are computed once for the whole raster.
    """
    valid = src.read(VALID_BAND_INDEX) >= 1
    limits: dict[str, tuple[float, float]] = {}
    for name in RGB:
        band = src.read(SPECTRAL_BANDS.index(name) + 1)
        values = band[valid]
        del band
        if values.size == 0:
            raise ValueError("no valid pixel in the raster: cannot compute the preview stretch")
        lo, hi = np.percentile(values, STRETCH_PERCENTILES)
        limits[name] = (float(lo), float(hi))
    return limits


def write_patch(path: Path, data: np.ndarray, transform: Affine) -> None:
    """Write a 13 x 64 x 64 uint16 GeoTIFF (EPSG:32613, deflate, predictor 2, band descriptions)."""
    import rasterio

    if data.shape != (N_SPECTRAL, CELL_PIXELS, CELL_PIXELS) or data.dtype != np.uint16:
        raise ValueError(
            f"expected uint16 {(N_SPECTRAL, CELL_PIXELS, CELL_PIXELS)}, "
            f"got {data.dtype} {data.shape}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=CELL_PIXELS,
        height=CELL_PIXELS,
        count=N_SPECTRAL,
        dtype="uint16",
        crs=EXPORT_CRS,
        transform=transform,
        compress="deflate",
        predictor=2,
    ) as dst:
        dst.write(data)
        for i, name in enumerate(SPECTRAL_BANDS, start=1):
            dst.set_band_description(i, name)


def write_preview_png(path: Path, image: np.ndarray) -> None:
    """Write a uint8 RGB image as a 3-channel PNG (no alpha channel)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def cell_centroids_lonlat(grid: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cell centroids ``(lat, lon)`` in EPSG:4326."""
    points = grid.geometry.centroid.set_crs(grid.crs, allow_override=True).to_crs("EPSG:4326")
    return points.y.to_numpy(), points.x.to_numpy()


def tile(
    vrt: Path,
    grid_gpkg: Path,
    out_dir: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    """Tile the raster into patches and previews and write ``cells.csv``.

    Args:
        vrt: 14-band export raster (13 spectral bands + ``valid_obs``).
        grid_gpkg: grid file written by :mod:`hydroscope.geo.grid` (layer ``grid``).
        out_dir: output root (``patches/``, ``previews/`` and ``cells.csv`` are created inside).
        limit: process only the first ``limit`` cells (quick dry check).

    Returns:
        The cells index (columns :data:`CELLS_COLUMNS`), also written to ``cells.csv``.

    Raises:
        ValueError: if the raster has the wrong band count or a cell is not on the pixel lattice.
    """
    import rasterio

    grid = gpd.read_file(grid_gpkg, layer="grid").sort_values(["row", "col"]).reset_index(drop=True)
    if limit is not None:
        grid = grid.head(limit)
    lat, lon = cell_centroids_lonlat(grid)

    patch_dir, preview_dir = out_dir / "patches", out_dir / "previews"
    records: list[dict[str, Any]] = []
    with rasterio.open(vrt) as src:
        if src.count != VALID_BAND_INDEX:
            raise ValueError(f"expected {VALID_BAND_INDEX} bands, found {src.count}")
        limits = rgb_stretch_limits(src)
        preview_dir.mkdir(parents=True, exist_ok=True)
        (preview_dir / "stretch_rgb.json").write_text(
            json.dumps({k: list(v) for k, v in limits.items()}, indent=2)
        )
        rgb_idx = [SPECTRAL_BANDS.index(b) for b in RGB]

        for _, group in grid.groupby("row", sort=True):
            bounds = group.geometry.bounds
            strip = None
            for idx, cell in group.iterrows():
                minx, maxy = float(bounds.at[idx, "minx"]), float(bounds.at[idx, "maxy"])
                col_off, row_off = pixel_offset(minx, maxy, src.transform)
                if strip is None:  # all cells of a grid row share the same row offset
                    strip = read_strip(src, row_off)
                data = extract_window(strip, col_off)
                vfrac = valid_fraction(data[VALID_BAND_INDEX - 1])
                frac = float(cell["frac_in_state"])
                has_patch = passes_filters(frac, vfrac)
                if has_patch:
                    transform = Affine(PIXEL_SIZE_M, 0.0, minx, 0.0, -PIXEL_SIZE_M, maxy)
                    write_patch(patch_dir / f"{cell['cell_id']}.tif", data[:N_SPECTRAL], transform)
                    write_preview_png(
                        preview_dir / f"{cell['cell_id']}.png",
                        render_preview(data[rgb_idx], limits),
                    )
                records.append(
                    {
                        "cell_id": cell["cell_id"],
                        "block_id": cell["block_id"],
                        "frac_in_state": frac,
                        "valid_frac": vfrac,
                        "lat": float(lat[idx]),
                        "lon": float(lon[idx]),
                        "has_patch": has_patch,
                    }
                )

    cells = (
        pd.DataFrame(records, columns=CELLS_COLUMNS).sort_values("cell_id").reset_index(drop=True)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out_dir / "cells.csv", index=False)
    return cells


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m hydroscope.geo.tiling",
        description="Tile the Aguascalientes export into 13x64x64 patches and RGB previews.",
    )
    parser.add_argument("--vrt", type=Path, default=Path("data/ags/raw/s2_ags_2024_04.vrt"))
    parser.add_argument("--grid", type=Path, default=Path("data/ref/grid.gpkg"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/ags"))
    parser.add_argument("--limit", type=int, default=None, help="process only the first N cells")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    for path in (args.vrt, args.grid):
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1
    try:
        cells = tile(args.vrt, args.grid, args.out_dir, args.limit)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{len(cells)} cells, {int(cells['has_patch'].sum())} patches -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
