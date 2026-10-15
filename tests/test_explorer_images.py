"""scripts/make_explorer_images.py: sprites and images.js (no EuroSAT, patches or checkpoints)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# make_explorer_images imports make_gradcam, so the scripts folder must be importable.
sys.path.insert(0, str(ROOT / "scripts"))
import make_explorer_images as mei  # noqa: E402

RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def solid_tile(color: tuple[int, int, int]) -> np.ndarray:
    """A (64, 64, 3) image in [0, 1] of one color."""
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[:, :] = np.array(color) / 255
    return image


def read_meta(js_path: Path) -> dict:
    """The JSON of images.js, after checking its two lines."""
    lines = js_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == mei.HEADER
    assert lines[1].startswith(mei.PREFIX)
    assert lines[1].endswith(";")
    return json.loads(lines[1].removeprefix(mei.PREFIX).removesuffix(";"))


def test_tiles_fill_rows_of_16():
    assert mei.tile_position(0) == (0, 0)
    assert mei.tile_position(15) == (960, 0)
    assert mei.tile_position(16) == (0, 64)
    assert mei.tile_position(17) == (64, 64)


def test_each_tile_lands_at_its_position_in_the_sprite():
    tiles = []
    positions = []
    for k in range(17):
        color = BLUE if k == 16 else RED
        positions.append(mei.add_tile(tiles, solid_tile(color)))
    sprite = mei.build_sprite(tiles)
    assert sprite.size == (1024, 128)  # 16 tiles per row, 2 rows
    assert positions[16] == {"x": 0, "y": 64}
    assert sprite.getpixel((32, 96)) == BLUE  # centre of tile 16
    assert sprite.getpixel((15 * 64 + 32, 32)) == RED  # centre of tile 15
    assert sprite.getpixel((64 + 32, 96)) == WHITE  # tile 17 does not exist


def test_fewer_patches_than_per_class_are_all_used():
    labels = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # 2 AnnualCrop, 8 Forest, no other class
    chosen = mei.pick_per_class(labels, 6, 0)
    assert sorted(chosen["AnnualCrop"]) == [0, 1]
    assert len(chosen["Forest"]) == 6
    assert chosen["SeaLake"] == []


def test_images_js_has_the_header_and_valid_json():
    meta = {"schema_version": 1, "tile": 64, "samples": None, "gradcam": None}
    text = mei.images_js_text(meta)
    assert text.startswith(mei.HEADER + "\n" + mei.PREFIX)
    assert json.loads(text.splitlines()[1].removeprefix(mei.PREFIX).removesuffix(";")) == meta


def test_without_inputs_samples_and_gradcam_are_null(tmp_path):
    written = mei.write_images(
        eurosat_cache=tmp_path / "cache",
        labels_csv=tmp_path / "labels.csv",
        patches_dir=tmp_path / "patches",
        gradcam_csv=tmp_path / "gradcam_samples.csv",
        out_dir=tmp_path / "out",
    )
    assert [path.name for path in written] == ["images.js"]
    meta = read_meta(tmp_path / "out" / "images.js")
    assert meta == {"schema_version": 1, "tile": 64, "samples": None, "gradcam": None}
