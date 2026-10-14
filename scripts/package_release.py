"""Build the v1.0 release assets in dist/. Nothing is uploaded.

Run: python scripts/package_release.py   (or: make release)

Assets written to dist/:
- ags_labeled_v1.0.zip: data/ags/labels.csv, configs/ags_block_split.csv, the labeled patches
  of data/ags/patches (cells labeled "skip" are left out) and data/ags/NOTICE.
- the 24 checkpoints (.pt) with their run config (.yaml), as separate files.
- RELEASE_NOTES.md: what each asset holds, the commit and the g3/g4 tags.
- SHA256SUMS.txt: checksum of every other asset (`shasum -a 256 -c SHA256SUMS.txt` checks it).
Run it on a clean, committed tree, so the commit in the notes is the one that is released.
"""

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

from hydroscope.data.labels import SKIP, load_labels

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
CHECKPOINT_DIR = ROOT / "checkpoints"
LABELS_CSV = ROOT / "data" / "ags" / "labels.csv"
PATCHES_DIR = ROOT / "data" / "ags" / "patches"
SPLIT_CSV = ROOT / "configs" / "ags_block_split.csv"

TAG = "v1.0"
ZIP_NAME = f"ags_labeled_{TAG}.zip"
NOTES_NAME = "RELEASE_NOTES.md"
CHECKSUMS_NAME = "SHA256SUMS.txt"
N_CHECKPOINTS = 24  # 8 EuroSAT runs x (EuroSAT, ft-head, ft-last_block)
REPO_URL = "https://github.com/alejandro-serrano/ags-hydroscope"

# Git tags named in the release notes, with what each one marks.
GIT_TAGS = {
    "g3-results-frozen": "results frozen (G3 checkpoint)",
    "g4-full-draft": "full paper draft (G4 checkpoint)",
}

NOTICE = f"""Hydroscope Aguascalientes - labeled patches {TAG}

Contains modified Copernicus Sentinel-2 data (2024).

The labels (data/ags/labels.csv) and the spatial split (configs/ags_block_split.csv) were made
by the author of the repository and are released under its license (MIT):
{REPO_URL}
"""


def git_output(args: list[str]) -> str:
    """Run a git command in the repo and return its output without the final newline."""
    output = subprocess.run(["git"] + args, capture_output=True, text=True, check=True, cwd=ROOT)
    return output.stdout.strip()


def labeled_patches(labels_csv: Path, patches_dir: Path) -> list[Path]:
    """Patch files of every labeled cell, without the cells labeled skip."""
    labels = load_labels(labels_csv)
    labeled = labels[labels["label"] != SKIP]
    paths = []
    missing = []
    for cell_id in sorted(labeled["cell_id"]):
        path = patches_dir / f"{cell_id}.tif"
        if path.exists():
            paths.append(path)
        else:
            missing.append(path.name)
    if missing:
        raise ValueError(f"{len(missing)} labeled patches are missing, e.g. {missing[:3]}")
    return paths


def make_zip(patch_paths: list[Path], zip_path: Path) -> None:
    """Write the zip with paths relative to the repo root, so it unpacks in place."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(LABELS_CSV, LABELS_CSV.relative_to(ROOT))
        archive.write(SPLIT_CSV, SPLIT_CSV.relative_to(ROOT))
        archive.writestr("data/ags/NOTICE", NOTICE)
        for path in patch_paths:
            archive.write(path, path.relative_to(ROOT))


def checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    """Every .pt with its .yaml, in name order. Fails if a .yaml is missing or the count is off."""
    files = []
    for pt_path in sorted(checkpoint_dir.glob("*.pt")):
        yaml_path = pt_path.with_suffix(".yaml")
        if not yaml_path.exists():
            raise ValueError(f"missing run config {yaml_path.name} for {pt_path.name}")
        files += [pt_path, yaml_path]
    if len(files) != 2 * N_CHECKPOINTS:
        raise ValueError(f"expected {N_CHECKPOINTS} checkpoints, found {len(files) // 2}")
    return files


def run_ids(checkpoint_paths: list[Path]) -> list[str]:
    """The EuroSAT run ids (checkpoints whose name has no _ft- part)."""
    ids = []
    for path in checkpoint_paths:
        if path.suffix == ".pt" and "_ft-" not in path.stem:
            ids.append(path.stem)
    return ids


def release_notes(n_patches: int, ids: list[str], commit: str, dirty: bool) -> str:
    """Text of RELEASE_NOTES.md."""
    lines = [f"# Hydroscope Aguascalientes {TAG}", ""]
    lines.append(
        "Data and checkpoints of the stage-1 study: two CNNs and a Vision Transformer trained on "
        "EuroSAT and tested on Sentinel-2 patches of Aguascalientes, Mexico."
    )
    lines.append("")
    lines.append(f"- Built by `scripts/package_release.py` from commit `{commit}`.")
    if dirty:
        lines.append("- WARNING: the working tree had uncommitted changes when this was built.")
    for tag, meaning in GIT_TAGS.items():
        tag_commit = git_output(["rev-list", "-n", "1", tag])
        lines.append(f"- Tag `{tag}`: {meaning}, commit `{tag_commit}`.")
    lines.append("- Download and check everything with `make download` (from the repo root).")
    lines.append("")
    lines.append("## Assets")
    lines.append("")
    lines.append("| Asset | Contents |")
    lines.append("| --- | --- |")
    lines.append(
        f"| `{ZIP_NAME}` | {n_patches} labeled Aguascalientes patches "
        "(`data/ags/patches/<cell_id>.tif`, GeoTIFF 13x64x64 uint16, cells labeled skip left out), "
        "`data/ags/labels.csv`, `configs/ags_block_split.csv` and `data/ags/NOTICE` |"
    )
    lines.append(
        "| `<run_id>.pt` + `.yaml` | Weights (`state_dict`) trained on EuroSAT and the run config |"
    )
    lines.append(
        "| `<run_id>_ft-head.pt` + `.yaml` | The same model fine-tuned on Aguascalientes, "
        "head only |"
    )
    lines.append(
        "| `<run_id>_ft-last_block.pt` + `.yaml` | The same model fine-tuned on Aguascalientes, "
        "last block and head |"
    )
    lines.append(f"| `{NOTES_NAME}` | This file |")
    lines.append(f"| `{CHECKSUMS_NAME}` | SHA256 of every other asset |")
    lines.append("")
    lines.append("Run ids (`<model>_<bands>_<weights>`):")
    lines.append("")
    for run_id in ids:
        lines.append(f"- `{run_id}`")
    lines.append("")
    lines.append("## Licenses")
    lines.append("")
    lines.append(
        "Code, labels and checkpoints: MIT (see `LICENSE` in the repository). The patches contain "
        "modified Copernicus Sentinel-2 data (2024). See the README for the other data sources "
        "and pretrained weights."
    )
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    """SHA256 of a file, as hex text."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checksums(paths: list[Path], out_path: Path) -> None:
    """One line per file, in the format of `shasum -a 256`."""
    lines = []
    for path in paths:
        lines.append(f"{sha256(path)}  {path.name}")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    DIST_DIR.mkdir(exist_ok=True)
    commit = git_output(["rev-parse", "HEAD"])
    dirty = git_output(["status", "--porcelain", "--untracked-files=no"]) != ""
    if dirty:
        print("WARNING: uncommitted changes; commit first so the notes name the right commit")

    patch_paths = labeled_patches(LABELS_CSV, PATCHES_DIR)
    zip_path = DIST_DIR / ZIP_NAME
    print(f"zipping {len(patch_paths)} labeled patches -> {zip_path.relative_to(ROOT)}")
    make_zip(patch_paths, zip_path)

    assets = [zip_path]
    checkpoints = checkpoint_files(CHECKPOINT_DIR)
    print(f"copying {len(checkpoints) // 2} checkpoints (.pt + .yaml) to dist/")
    for path in checkpoints:
        shutil.copy2(path, DIST_DIR / path.name)
        assets.append(DIST_DIR / path.name)

    notes_path = DIST_DIR / NOTES_NAME
    notes_path.write_text(release_notes(len(patch_paths), run_ids(checkpoints), commit, dirty))
    assets.append(notes_path)

    checksums_path = DIST_DIR / CHECKSUMS_NAME
    write_checksums(assets, checksums_path)
    assets.append(checksums_path)

    total = 0
    print(f"\nRelease {TAG} assets in dist/ (commit {commit[:7]}):")
    for path in assets:
        size = path.stat().st_size
        total += size
        print(f"  {path.name:<48} {size / 1e6:9.1f} MB")
    print(f"  {'total (' + str(len(assets)) + ' files)':<48} {total / 1e6:9.1f} MB")


if __name__ == "__main__":
    main()
