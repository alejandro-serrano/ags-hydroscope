"""Download the v1.0 release assets, check their SHA256 and put them in place.

Run from the repo root: python scripts/download_release.py   (or: make download)

Uses `gh release download` if the GitHub CLI is installed, otherwise urllib (the repository is
public, so no login is needed). The files first go to dist/v1.0/. Only when every checksum
matches are the checkpoints (.pt + .yaml) moved to checkpoints/ and the zip unpacked into
data/ags/ (labels.csv, patches/ and NOTICE). Existing files with the same name are replaced.
"""

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "alejandro-serrano/ags-hydroscope"
TAG = "v1.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"
DOWNLOAD_DIR = ROOT / "dist" / TAG
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKSUMS_NAME = "SHA256SUMS.txt"
ZIP_NAME = f"ags_labeled_{TAG}.zip"
DATA_PREFIX = "data/ags/"  # only these zip members are unpacked; the split CSV is already in git


def download_with_gh(out_dir: Path) -> None:
    """Download every asset of the release with the GitHub CLI."""
    command = ["gh", "release", "download", TAG, "--repo", REPO, "--dir", str(out_dir)]
    subprocess.run(command + ["--clobber"], check=True)


def download_file(name: str, out_dir: Path) -> None:
    """Download one asset with urllib."""
    print(f"downloading {name}")
    urllib.request.urlretrieve(f"{BASE_URL}/{name}", out_dir / name)


def read_checksums(path: Path) -> dict[str, str]:
    """{file name: sha256} from a SHA256SUMS.txt written by scripts/package_release.py."""
    checksums = {}
    for line in path.read_text().splitlines():
        if line.strip() != "":
            digest, name = line.split("  ", 1)
            checksums[name] = digest
    return checksums


def download_with_urllib(out_dir: Path) -> None:
    """Download SHA256SUMS.txt, then every file it lists."""
    download_file(CHECKSUMS_NAME, out_dir)
    for name in read_checksums(out_dir / CHECKSUMS_NAME):
        download_file(name, out_dir)


def sha256(path: Path) -> str:
    """SHA256 of a file, as hex text."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(out_dir: Path, checksums: dict[str, str]) -> None:
    """Fail with the list of files that are missing or whose SHA256 does not match."""
    bad = []
    for name, digest in checksums.items():
        path = out_dir / name
        if not path.exists():
            bad.append(f"{name} (missing)")
        elif sha256(path) != digest:
            bad.append(f"{name} (SHA256 mismatch)")
    if bad:
        raise ValueError("release check failed: " + ", ".join(bad))
    print(f"SHA256 OK for {len(checksums)} files")


def install(out_dir: Path, checksums: dict[str, str]) -> None:
    """Move the checkpoints to checkpoints/ and unpack the data part of the zip."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    n_moved = 0
    for name in checksums:
        if name.endswith(".pt") or name.endswith(".yaml"):
            shutil.move(out_dir / name, CHECKPOINT_DIR / name)
            n_moved += 1
    print(f"moved {n_moved} checkpoint files to checkpoints/")

    with zipfile.ZipFile(out_dir / ZIP_NAME) as archive:
        members = []
        for name in archive.namelist():
            if name.startswith(DATA_PREFIX):
                members.append(name)
        archive.extractall(ROOT, members=members)
    print(f"unpacked {len(members)} files into {DATA_PREFIX}")


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.which("gh") is not None:
        download_with_gh(DOWNLOAD_DIR)
    else:
        print("gh not found, downloading with urllib")
        download_with_urllib(DOWNLOAD_DIR)

    checksums = read_checksums(DOWNLOAD_DIR / CHECKSUMS_NAME)
    check_files(DOWNLOAD_DIR, checksums)
    install(DOWNLOAD_DIR, checksums)


if __name__ == "__main__":
    main()
