"""scripts/make_explorer_basemap.py: mosaic of the cell previews (no data/ags needed)."""

import importlib.util
import json
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_explorer_basemap.py"
spec = importlib.util.spec_from_file_location("make_explorer_basemap", SCRIPT)
meb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meb)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def write_previews(previews_dir: Path) -> None:
    """Three tiny opaque previews: red at (5, 10), green at (5, 12), blue at (7, 10)."""
    previews_dir.mkdir()
    cells = {"r0005_c0010": RED, "r0005_c0012": GREEN, "r0007_c0010": BLUE}
    for cell_id, color in cells.items():
        Image.new("RGBA", (16, 16), color + (255,)).save(previews_dir / f"{cell_id}.png")


def test_each_color_lands_on_its_cell(tmp_path):
    write_previews(tmp_path / "previews")
    paths = meb.preview_paths(tmp_path / "previews")
    extent = meb.basemap_extent([meb.parse_cell_id(path.stem) for path in paths])
    mosaic = meb.build_mosaic(paths, extent)
    assert mosaic.size == (24, 24)  # 3 x 3 cells of 8 px
    # Centre pixel of each 8 x 8 block: (x, y) = (col * 8 + 4, row * 8 + 4), relative to row0/col0
    assert mosaic.getpixel((4, 4)) == RED
    assert mosaic.getpixel((20, 4)) == GREEN
    assert mosaic.getpixel((4, 20)) == BLUE
    assert mosaic.getpixel((12, 12)) == WHITE  # cell (6, 11) has no preview


def test_extent_starts_at_the_first_row_and_column(tmp_path):
    write_previews(tmp_path / "previews")
    paths = meb.preview_paths(tmp_path / "previews")
    extent = meb.basemap_extent([meb.parse_cell_id(path.stem) for path in paths])
    assert extent == {"row0": 5, "col0": 10, "n_rows": 3, "n_cols": 3}


def test_basemap_js_has_the_header_and_valid_json(tmp_path):
    write_previews(tmp_path / "previews")
    written = meb.write_basemap(tmp_path / "previews", tmp_path / "out")
    assert [path.name for path in written] == ["basemap.jpg", "basemap.js"]
    lines = (tmp_path / "out" / "basemap.js").read_text().splitlines()
    assert lines[0] == meb.HEADER
    meta = json.loads(lines[1].removeprefix(meb.PREFIX).removesuffix(";"))
    assert meta == {
        "schema_version": 1,
        "file": "basemap.jpg",
        "row0": 5,
        "col0": 10,
        "n_rows": 3,
        "n_cols": 3,
        "pixels_per_cell": 8,
    }


def test_without_previews_nothing_is_written(tmp_path):
    assert meb.write_basemap(tmp_path / "previews", tmp_path / "out") == []
    assert not (tmp_path / "out").exists()
