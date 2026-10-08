"""Earth Engine export of the April 2024 Sentinel-2 L1C composite of Aguascalientes.

Usage::

    export EE_PROJECT=<your Earth Engine cloud project>   # never stored in code or in git
    python -m hydroscope.geo.gee_export --dry-run          # checks and parameters, no task started
    python -m hydroscope.geo.gee_export                    # starts the Drive export task
    python -m hydroscope.geo.gee_export --boundary data/ref/<inegi_state_file>
    python -m hydroscope.geo.gee_export --save-boundary data/ref/ags_state.gpkg   # no export task

What is exported (``COPERNICUS/S2_HARMONIZED``, level L1C, DN = reflectance x 10000):

* the images of ``2024-04-01 <= t < 2024-05-01`` intersecting the state;
* clouds masked with QA60 bit 10 (opaque) and bit 11 (cirrus);
* the per-pixel median of the 13 bands, renamed and ordered like TorchGeo EuroSAT
  (B01 ... B12, B8A last) and cast to ``uint16``;
* band 14 ``valid_obs``: per-pixel count of cloud-free observations (``uint16``);
* clipped to the state of Aguascalientes. Masked pixels (outside the state, or without any
  cloud-free observation) are written as 0 by Earth Engine, so ``valid_obs == 0`` marks them.

Grid alignment: the export uses ``crsTransform = [10, 0, x0, 0, -10, y0]`` in EPSG:32613 with
``x0`` and ``y0`` snapped to multiples of 640 m. Pixels are 10 m, but the origin lies on the 640 m
lattice, so every 640 m grid cell of F2-03 is exactly one 64 x 64 pixel window.

The boundary is the GAUL level-1 feature (Mexico, Aguascalientes) unless ``--boundary`` gives an
INEGI file (read with geopandas; place it in ``data/ref/``, which is git-ignored).

``ee`` is imported lazily, so the pure helpers can be used and tested without Earth Engine.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

GAUL_ASSET = "FAO/GAUL/2025/level1"
GAUL_COUNTRY_PROP = "GAUL0_NAME"
GAUL_STATE_PROP = "GAUL1_NAME"
COUNTRY = "Mexico"
STATE = "Aguascalientes"

COLLECTION = "COPERNICUS/S2_HARMONIZED"
START_DATE = "2024-04-01"
END_DATE = "2024-05-01"
"""Half-open interval: ``START_DATE <= t < END_DATE`` (``filterDate`` excludes the end)."""

DRIVE_FOLDER = "ags-hydroscope"
FILE_PREFIX = "s2_ags_2024_04"
EXPORT_CRS = "EPSG:32613"
PIXEL_SIZE_M = 10
GRID_M = 640
MAX_PIXELS = 10_000_000_000
ENV_VAR = "EE_PROJECT"
BOUNDARY_LAYER = "state"

QA60_OPAQUE_BIT = 10
QA60_CIRRUS_BIT = 11
QA60_OPAQUE_MASK = 1 << QA60_OPAQUE_BIT
QA60_CIRRUS_MASK = 1 << QA60_CIRRUS_BIT
QA60_CLOUD_MASK = QA60_OPAQUE_MASK | QA60_CIRRUS_MASK

VALID_OBS_BAND = "valid_obs"
CLOUD_PCT_PROPERTY = "CLOUDY_PIXEL_PERCENTAGE"
QA60_DIAGNOSTIC_SCALE_M = 200
"""Coarse scale (m) of the QA60-flagged fraction diagnostic; QA60 is a 60 m band."""
CLOUD_WARN_PERCENT = 1.0

GEE_TO_EUROSAT_BANDS: tuple[tuple[str, str], ...] = (
    ("B1", "B01"),
    ("B2", "B02"),
    ("B3", "B03"),
    ("B4", "B04"),
    ("B5", "B05"),
    ("B6", "B06"),
    ("B7", "B07"),
    ("B8", "B08"),
    ("B9", "B09"),
    ("B10", "B10"),
    ("B11", "B11"),
    ("B12", "B12"),
    ("B8A", "B8A"),
)
"""(Earth Engine band name, EuroSAT band name) in TorchGeo EuroSAT order, B8A last."""


class MissingProjectError(RuntimeError):
    """Raised when the ``EE_PROJECT`` environment variable is not set."""


# --------------------------------------------------------------------------------------------
# Pure helpers (no Earth Engine)
# --------------------------------------------------------------------------------------------


def gee_band_names() -> list[str]:
    """Earth Engine band names in export order (B1 ... B12, B8A last)."""
    return [gee for gee, _ in GEE_TO_EUROSAT_BANDS]


def eurosat_band_names() -> list[str]:
    """EuroSAT band names in export order (B01 ... B12, B8A last)."""
    return [euro for _, euro in GEE_TO_EUROSAT_BANDS]


def band_mapping() -> dict[str, str]:
    """Mapping Earth Engine band name -> EuroSAT band name."""
    return dict(GEE_TO_EUROSAT_BANDS)


def export_band_names() -> list[str]:
    """Band names of the exported image: the 13 EuroSAT bands followed by ``valid_obs``."""
    return [*eurosat_band_names(), VALID_OBS_BAND]


def is_cloudy(qa60: int) -> bool:
    """True if QA60 bit 10 (opaque clouds) or bit 11 (cirrus) is set."""
    return bool(int(qa60) & QA60_CLOUD_MASK)


def snap_origin(
    x_min: float, y_max: float, grid: int = GRID_M, pixel: int = PIXEL_SIZE_M
) -> tuple[int, int]:
    """Snap the north-west corner of a bounding box outward to the ``grid``-metre lattice.

    Args:
        x_min: western edge of the bounding box (m).
        y_max: northern edge of the bounding box (m).
        grid: lattice spacing in metres (must be a multiple of ``pixel``).
        pixel: pixel size in metres.

    Returns:
        ``(x0, y0)``: ``x0 = floor(x_min / grid) * grid`` and ``y0 = ceil(y_max / grid) * grid``,
        so the snapped box always contains the original one.
    """
    if grid % pixel != 0:
        raise ValueError(f"grid ({grid} m) must be a multiple of the pixel size ({pixel} m)")
    return math.floor(x_min / grid) * grid, math.ceil(y_max / grid) * grid


def snap_extent(
    bounds: Sequence[float], grid: int = GRID_M, pixel: int = PIXEL_SIZE_M
) -> tuple[int, int, int, int]:
    """Snap a bounding box ``(x_min, y_min, x_max, y_max)`` outward to the ``grid`` lattice.

    Returns:
        ``(x0, y0, x1, y1)`` with ``x0, y0`` the north-west corner (``y0`` is the northern edge)
        and ``x1, y1`` the south-east corner (``y1`` is the southern edge), all multiples of
        ``grid``.
    """
    x_min, y_min, x_max, y_max = bounds
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"invalid bounds {tuple(bounds)}")
    x0, y0 = snap_origin(x_min, y_max, grid, pixel)
    x1 = math.ceil(x_max / grid) * grid
    y1 = math.floor(y_min / grid) * grid
    return x0, y0, x1, y1


def crs_transform(x0: float, y0: float, pixel: int = PIXEL_SIZE_M) -> list[float]:
    """Affine ``crsTransform`` ``[pixel, 0, x0, 0, -pixel, y0]`` (north-up, origin at x0, y0)."""
    return [pixel, 0, x0, 0, -pixel, y0]


def raster_size(extent: Sequence[int], pixel: int = PIXEL_SIZE_M) -> tuple[int, int]:
    """Width and height in pixels of a snapped extent ``(x0, y0, x1, y1)``."""
    x0, y0, x1, y1 = extent
    return (x1 - x0) // pixel, (y0 - y1) // pixel


def qa60_mask_is_noop(flagged_fraction: float | None, mean_cloud_percent: float | None) -> bool:
    """True if the QA60 mask looks like a no-op: nothing flagged although scenes report clouds.

    ``None`` values mean the diagnostic could not be computed and give ``False`` (no verdict).
    """
    if flagged_fraction is None or mean_cloud_percent is None:
        return False
    return flagged_fraction == 0 and mean_cloud_percent > CLOUD_WARN_PERCENT


def get_project(environ: Mapping[str, str] | None = None) -> str:
    """Return the Earth Engine project from ``EE_PROJECT``.

    Raises:
        MissingProjectError: if the variable is unset or empty. The message never contains a
            project identifier.
    """
    env = os.environ if environ is None else environ
    project = env.get(ENV_VAR, "").strip()
    if not project:
        raise MissingProjectError(
            f"environment variable {ENV_VAR} is not set. Export your Earth Engine cloud project "
            f"first, e.g. `export {ENV_VAR}=<your-project-id>`."
        )
    return project


def export_parameters(extent: Sequence[int]) -> dict[str, Any]:
    """Export parameters (without the image and region objects) for a snapped extent."""
    x0, y0, _, _ = extent
    width, height = raster_size(extent)
    return {
        "description": FILE_PREFIX,
        "folder": DRIVE_FOLDER,
        "fileNamePrefix": FILE_PREFIX,
        "crs": EXPORT_CRS,
        "crsTransform": crs_transform(x0, y0),
        "maxPixels": MAX_PIXELS,
        "fileFormat": "GeoTIFF",
        "width_px": width,
        "height_px": height,
        "bands": export_band_names(),
    }


# --------------------------------------------------------------------------------------------
# Earth Engine part (lazy import of ee)
# --------------------------------------------------------------------------------------------


def initialize_ee(project: str) -> Any:
    """Import ``ee`` lazily and initialize it with ``project``; returns the module."""
    import ee

    ee.Initialize(project=project)
    return ee


def _gaul_states(ee: Any) -> Any:
    return ee.FeatureCollection(GAUL_ASSET).filter(ee.Filter.eq(GAUL_COUNTRY_PROP, COUNTRY))


def get_state_geometry(ee: Any, boundary: Path | None = None) -> Any:
    """Return the state geometry (``ee.Geometry``, EPSG:4326).

    Uses the GAUL feature with ``GAUL0_NAME == "Mexico"`` and ``GAUL1_NAME == "Aguascalientes"``
    (case-sensitive), or the INEGI file ``boundary`` when given.

    Raises:
        SystemExit: with a non-zero code if GAUL does not return exactly one feature; the sorted
            ``GAUL1_NAME`` values of Mexico are printed to help fixing the constants.
    """
    if boundary is not None:
        return _geometry_from_file(ee, boundary)
    states = _gaul_states(ee)
    selected = states.filter(ee.Filter.eq(GAUL_STATE_PROP, STATE))
    count = selected.size().getInfo()
    if count != 1:
        names = sorted(states.aggregate_array(GAUL_STATE_PROP).distinct().getInfo())
        print(
            f"ERROR: expected exactly 1 feature with {GAUL_COUNTRY_PROP}={COUNTRY!r} and "
            f"{GAUL_STATE_PROP}={STATE!r} in {GAUL_ASSET}, found {count}.",
            file=sys.stderr,
        )
        print(f"{GAUL_STATE_PROP} values for {COUNTRY}:", file=sys.stderr)
        for name in names:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)
    return selected.geometry()


def boundary_frame(geojson: Mapping[str, Any]) -> Any:
    """Convert a GeoJSON geometry (EPSG:4326) into a one-row ``GeoDataFrame``.

    Args:
        geojson: GeoJSON geometry mapping, e.g. ``ee.Geometry.getInfo()``. A ``Feature`` is also
            accepted (its ``geometry`` member is used).

    Returns:
        ``GeoDataFrame`` in EPSG:4326 with a ``name`` column (the state name) and the geometry.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    geometry = geojson.get("geometry", geojson) if geojson.get("type") == "Feature" else geojson
    return gpd.GeoDataFrame({"name": [STATE]}, geometry=[shape(geometry)], crs="EPSG:4326")


def save_boundary(ee: Any, geometry: Any, path: Path) -> Path:
    """Fetch ``geometry`` from Earth Engine and write it to a GPKG (layer ``state``)."""
    frame = boundary_frame(geometry.getInfo())
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GPKG", layer=BOUNDARY_LAYER)
    return path


def _geometry_from_file(ee: Any, path: Path) -> Any:
    """Read a vector file, dissolve it into one EPSG:4326 geometry and convert it to ee."""
    import geopandas as gpd
    from shapely.geometry import mapping

    if not path.exists():
        raise FileNotFoundError(f"boundary file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"boundary file has no features: {path}")
    gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    return ee.Geometry(mapping(geom))


def ring_extent(ring: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """``(x_min, y_min, x_max, y_max)`` of a bounding-box ring in EPSG:32613 metres.

    Raises:
        ValueError: if the ring is empty or its coordinates look like degrees, i.e. the bounds
            were not computed in the projected CRS.
    """
    if not ring:
        raise ValueError("empty bounds ring")
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    extent = min(xs), min(ys), max(xs), max(ys)
    if max(abs(v) for v in extent) <= 360:
        raise ValueError(f"bounds look like degrees, expected {EXPORT_CRS} metres: {extent}")
    return extent


def state_bounds_utm(ee: Any, geometry: Any) -> tuple[float, float, float, float]:
    """Bounding box ``(x_min, y_min, x_max, y_max)`` of the geometry in EPSG:32613 (metres).

    The bounds are computed directly in the export CRS with a 1 m error margin; Earth Engine
    rejects ``bounds()`` of a reprojected geometry with the default zero margin.
    """
    box = geometry.bounds(maxError=1, proj=EXPORT_CRS)
    return ring_extent(box.coordinates().getInfo()[0])


def mask_s2_clouds(ee: Any, image: Any) -> Any:
    """Mask pixels with QA60 bit 10 (opaque clouds) or bit 11 (cirrus) set."""
    qa = image.select("QA60")
    clear = qa.bitwiseAnd(QA60_OPAQUE_MASK).eq(0).And(qa.bitwiseAnd(QA60_CIRRUS_MASK).eq(0))
    return image.updateMask(clear)


def filtered_collection(ee: Any, geometry: Any) -> Any:
    """Unmasked S2 L1C collection for the month and the state."""
    return ee.ImageCollection(COLLECTION).filterDate(START_DATE, END_DATE).filterBounds(geometry)


def build_composite(ee: Any, collection: Any, geometry: Any) -> Any:
    """Median composite (13 EuroSAT-ordered ``uint16`` bands) plus ``valid_obs``, clipped."""
    masked = collection.map(lambda img: mask_s2_clouds(ee, img))
    bands = masked.select(gee_band_names(), eurosat_band_names()).median().toUint16()
    valid_obs = masked.select("B2").count().rename(VALID_OBS_BAND).toUint16()
    return bands.addBands(valid_obs).select(export_band_names()).clip(geometry)


def collection_diagnostics(ee: Any, collection: Any, geometry: Any) -> dict[str, Any]:
    """Scene count, mean ``CLOUDY_PIXEL_PERCENTAGE`` and QA60-flagged pixel fraction.

    The flagged fraction is the mean over scenes and over the state (coarse scale) of the
    indicator ``QA60 & (bit 10 | bit 11) != 0``. ``None`` values mean the server returned nothing.
    """
    scenes = collection.size().getInfo()
    mean_cloud = collection.aggregate_mean(CLOUD_PCT_PROPERTY).getInfo() if scenes else None
    flagged_fraction = None
    if scenes:
        flagged = collection.map(
            lambda img: (
                img.select("QA60").bitwiseAnd(QA60_CLOUD_MASK).neq(0).rename("flagged").toFloat()
            )
        ).mean()
        result = flagged.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=QA60_DIAGNOSTIC_SCALE_M,
            maxPixels=10**9,
            bestEffort=True,
        ).getInfo()
        flagged_fraction = result.get("flagged") if result else None
    return {
        "scenes": scenes,
        "mean_cloud_percent": mean_cloud,
        "qa60_flagged_fraction": flagged_fraction,
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m hydroscope.geo.gee_export",
        description="Export the April 2024 Sentinel-2 L1C composite of Aguascalientes to Drive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="initialize Earth Engine and print checks and parameters, but never start the task",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=None,
        help="INEGI state boundary file (e.g. data/ref/<file>.gpkg) instead of FAO GAUL",
    )
    parser.add_argument(
        "--save-boundary",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the state geometry (GAUL, or --boundary) to a GPKG (layer 'state', "
        "EPSG:4326) and exit without starting an export task",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        project = get_project()
    except MissingProjectError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    ee = initialize_ee(project)

    geometry = get_state_geometry(ee, args.boundary)
    source = str(args.boundary) if args.boundary else f"{GAUL_ASSET} ({COUNTRY}, {STATE})"
    print(f"Boundary: {source}")

    if args.save_boundary is not None:
        save_boundary(ee, geometry, args.save_boundary)
        print(f"Wrote state boundary to {args.save_boundary} (layer '{BOUNDARY_LAYER}')")
        return 0

    collection = filtered_collection(ee, geometry)
    diag = collection_diagnostics(ee, collection, geometry)
    print(f"Collection: {COLLECTION}, {START_DATE} <= t < {END_DATE}")
    print(f"Scenes: {diag['scenes']}")
    print(f"Mean {CLOUD_PCT_PROPERTY}: {_fmt(diag['mean_cloud_percent'])}")
    print(f"QA60-flagged pixel fraction: {_fmt(diag['qa60_flagged_fraction'], 4)}")
    if diag["scenes"] == 0:
        print("ERROR: no scenes found for the state and period.", file=sys.stderr)
        return 1
    if qa60_mask_is_noop(diag["qa60_flagged_fraction"], diag["mean_cloud_percent"]):
        print(
            "WARNING: QA60 flags no pixel although scenes report clouds "
            f"(mean {CLOUD_PCT_PROPERTY} > {CLOUD_WARN_PERCENT}). QA60 may be unpopulated, so the "
            "cloud mask would be a no-op. The mask is NOT switched automatically; decide before "
            "exporting."
        )

    extent = snap_extent(state_bounds_utm(ee, geometry))
    params = export_parameters(extent)
    x0, y0, x1, y1 = extent
    print(f"Extent {EXPORT_CRS} (snapped to {GRID_M} m): x {x0}..{x1}, y {y1}..{y0}")
    print(f"Raster: {params['width_px']} x {params['height_px']} px at {PIXEL_SIZE_M} m")
    for key in ("folder", "fileNamePrefix", "crs", "crsTransform", "maxPixels", "bands"):
        print(f"{key}: {params[key]}")

    if args.dry_run:
        print("Dry run: no task started.")
        return 0

    image = build_composite(ee, collection, geometry)
    region = ee.Geometry.Rectangle([x0, y1, x1, y0], proj=EXPORT_CRS, geodesic=False)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=params["description"],
        folder=params["folder"],
        fileNamePrefix=params["fileNamePrefix"],
        region=region,
        crs=params["crs"],
        crsTransform=params["crsTransform"],
        maxPixels=params["maxPixels"],
        fileFormat=params["fileFormat"],
    )
    task.start()
    print(f"Task started: id={task.id}")
    print("Monitor: `earthengine task list` or the Tasks tab of the Earth Engine Code Editor.")
    print(f"When done, download the files from the Drive folder '{DRIVE_FOLDER}' to data/ags/raw/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
