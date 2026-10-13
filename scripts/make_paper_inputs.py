"""Make the inputs of the Data and Methodology sections of the paper.

Run: python scripts/make_paper_inputs.py

Outputs (nothing is typed by hand; every value is read from the repo):
- paper/tables/config.tex       shared configuration, read from configs/base.yaml
- paper/tables/ags_classes.tex  labeled patches per class and split, from data/ags/labels.csv
- paper/figures/study_area.pdf  (a) April 2024 RGB composite with the 8x8-cell blocks colored by
  split; (b) one EuroSAT and one Aguascalientes patch for three classes. The patches in (b) are
  the ones chosen with seed 0 for the Grad-CAM figures (paper/figures/gradcam_samples.csv), so
  run scripts/make_gradcam.py first. RGB stretch 2-98 % per dataset: Aguascalientes uses the
  global limits of the composite (the same as the labeling previews); EuroSAT uses the limits of
  the EuroSAT patches shown.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: we only save PDFs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from matplotlib.patches import Patch

from hydroscope.data.eurosat import CLASSES, EuroSATDataset
from hydroscope.data.labels import SKIP, load_labels
from hydroscope.data.transforms import SSL4EO_STD
from hydroscope.explain.gradcam import rgb_limits, to_rgb_image
from hydroscope.geo.tiling import RGB, SPECTRAL_BANDS, VALID_BAND_INDEX, rgb_stretch_limits
from hydroscope.training.evaluate import MIN_TEST_PATCHES

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base.yaml"
LABELS_CSV = ROOT / "data" / "ags" / "labels.csv"
PATCHES_DIR = ROOT / "data" / "ags" / "patches"
COMPOSITE = ROOT / "data" / "ags" / "raw" / "s2_ags_2024_04.tif"
GRID_GPKG = ROOT / "data" / "ref" / "grid.gpkg"
STATE_GPKG = ROOT / "data" / "ref" / "ags_state.gpkg"
BLOCK_SPLIT_CSV = ROOT / "configs" / "ags_block_split.csv"
SAMPLES_CSV = ROOT / "paper" / "figures" / "gradcam_samples.csv"
TABLES_DIR = ROOT / "paper" / "tables"
FIGURES_DIR = ROOT / "paper" / "figures"

MIN_FONT = 8
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts in the PDF
plt.rcParams["font.size"] = MIN_FONT

# Same colorblind-safe blue / orange pair as scripts/make_results.py.
SPLIT_COLORS = {"finetune": "#2a78d6", "test": "#eb6834"}
SPLIT_NAMES = {"finetune": "Fine-tune", "test": "Test"}
EXAMPLE_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation"]
BLOCK_ALPHA = 0.3  # light enough to see the composite under the blocks
DECIMATION = 8  # the composite is read at 1/8 of its resolution (80 m pixels) for panel (a)

NORMALIZATION_TEXT = {
    "eurosat_train": "per-band mean and std of EuroSAT train",
    "torchgeo": f"$x/\\num{{{SSL4EO_STD:.0f}}}$ (TorchGeo)",
}
AUGMENT_TEXT = {
    "hflip": "horizontal flip",
    "vflip": "vertical flip",
    "rot90": "90$^\\circ$ rotations",
}
INIT_TEXT = {"imagenet": "ImageNet", "ssl4eo": "SSL4EO-S12"}
LOSS_TEXT = {
    "cross_entropy": "cross-entropy",
    "class_weighted_cross_entropy": "class-weighted cross-entropy",
}
MONITOR_TEXT = {"val_macro_f1": "val macro-F1"}


# ---------------------------------------------------------------- Table: shared configuration


def num(value: int) -> str:
    """Count as siunitx \\num{}, so numbers of 4+ digits get a thousands separator."""
    return f"\\num{{{value}}}"


def tt(name: str) -> str:
    """Code name in typewriter font with LaTeX-safe underscores."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def config_rows(cfg: dict) -> list[tuple[str, str]]:
    """(setting, value) rows of the shared configuration table."""
    data = cfg["data"]
    optim = cfg["optim"]
    schedule = cfg["schedule"]
    stopping = cfg["early_stopping"]
    finetune = cfg["finetune"]

    size_in = f"{data['patch_size']}$\\times${data['patch_size']}"
    size_out = f"{data['input_size']}$\\times${data['input_size']}"
    norm_parts = []
    for init, method in data["normalization"].items():
        norm_parts.append(f"{INIT_TEXT[init]}: {NORMALIZATION_TEXT[method]}")
    augment_parts = []
    for name, enabled in data["augment"].items():
        if enabled:
            augment_parts.append(AUGMENT_TEXT[name])
    lr = f"\\num{{{optim['lr']:.0e}}}"  # 1e-04 -> \num{1e-04}, printed as 1x10^-4
    warmup = f"{schedule['warmup_epochs']} warmup epoch"
    if schedule["warmup_epochs"] != 1:
        warmup += "s"
    accum_steps = optim["batch_size"] // optim["micro_batch_size"]
    micro = f"{accum_steps}$\\times${optim['micro_batch_size']}"
    batch = f"{optim['batch_size']} ({micro}, accumulation)"
    monitor = MONITOR_TEXT[stopping["monitor"]]
    epochs = f"max {schedule['max_epochs']}, patience {stopping['patience']} on {monitor}"
    modes = ", ".join([tt(mode) for mode in finetune["modes"]])
    eurosat_loss = LOSS_TEXT[cfg["loss"]["eurosat"]]
    ags_loss = LOSS_TEXT[cfg["loss"]["ags"]]

    return [
        ("Input", f"{size_in} resized to {size_out} ({data['resize']})"),
        ("Normalization", "; ".join(norm_parts)),
        ("Augmentation", ", ".join(augment_parts)),
        ("Optimizer", f"AdamW, lr {lr}, weight decay {optim['weight_decay']}"),
        ("Schedule", f"{warmup} + {schedule['scheduler']}"),
        ("Batch size", batch),
        ("Epochs (EuroSAT)", epochs),
        ("Loss", f"EuroSAT: {eurosat_loss}; Aguascalientes: {ags_loss}"),
        ("Fine-tuning", f"modes {modes}; {finetune['epochs']} fixed epochs, last epoch kept"),
        ("Seed", f"{cfg['seed']}"),
    ]


def write_config_table(cfg: dict, path: Path) -> None:
    """LaTeX table of the configuration shared by every run."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Configuration shared by every run, from the base configuration file. Only the "
        "architecture, the input bands and the initial weights change between runs.}",
        "\\label{tab:config}",
        "\\begin{tabular}{@{}lp{0.68\\columnwidth}@{}}",
        "\\toprule",
        "Setting & Value \\\\",
        "\\midrule",
    ]
    for setting, value in config_rows(cfg):
        lines.append(f"{setting} & {value} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


# ---------------------------------------------------------------- Table: classes


def count_row(name: str, n_finetune: int, n_test: int) -> str:
    """One row of the class table: fine-tune, test and total counts."""
    return f"{name} & {num(n_finetune)} & {num(n_test)} & {num(n_finetune + n_test)} \\\\"


def write_classes_table(labels: pd.DataFrame, path: Path) -> None:
    """LaTeX table of labeled patches per class in the fine-tune and test splits."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Labeled Aguascalientes patches per class and split. "
        f"$^\\dagger$Fewer than {MIN_TEST_PATCHES} test patches: left out of the "
        "Aguascalientes macro-F1 and reported separately.}",
        "\\label{tab:ags_classes}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Class & Fine-tune & Test & Total \\\\",
        "\\midrule",
    ]
    finetune = labels[labels["split"] == "finetune"]
    test = labels[labels["split"] == "test"]
    for name in CLASSES:
        n_finetune = int((finetune["label"] == name).sum())
        n_test = int((test["label"] == name).sum())
        mark = "$^\\dagger$" if n_test < MIN_TEST_PATCHES else ""
        lines.append(count_row(f"{name}{mark}", n_finetune, n_test))
    lines.append("\\midrule")
    n_finetune = int((finetune["label"] != SKIP).sum())
    n_test = int((test["label"] != SKIP).sum())
    lines.append(count_row("Total labeled", n_finetune, n_test))
    n_finetune = int((finetune["label"] == SKIP).sum())
    n_test = int((test["label"] == SKIP).sum())
    lines.append(count_row("Skipped (not used)", n_finetune, n_test))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


# ---------------------------------------------------------------- Figure: study area


def read_composite_rgb(src, limits: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Decimated RGB of the composite in [0, 1] (white outside the data) and its extent."""
    height = src.height // DECIMATION
    width = src.width // DECIMATION
    image = np.ones((height, width, 3), dtype=np.float32)
    valid = src.read(VALID_BAND_INDEX, out_shape=(height, width)) >= 1  # (H, W)
    for i, name in enumerate(RGB):
        band = src.read(SPECTRAL_BANDS.index(name) + 1, out_shape=(height, width))
        low, high = limits[i]
        scaled = np.clip((band.astype(np.float32) - low) / (high - low), 0.0, 1.0)
        image[:, :, i] = np.where(valid, scaled, 1.0)
    left, bottom, right, top = src.bounds
    return image, [left, right, bottom, top]


def block_polygons(crs) -> gpd.GeoDataFrame:
    """One polygon per 8x8-cell block of the split, with its split name."""
    grid = gpd.read_file(GRID_GPKG).to_crs(crs)
    blocks = grid.dissolve(by="block_id").reset_index()[["block_id", "geometry"]]
    split = pd.read_csv(BLOCK_SPLIT_CSV)
    return blocks.merge(split, on="block_id")  # only blocks that belong to a split


def draw_map(ax, src, ags_limits: np.ndarray) -> None:
    """Panel (a): composite, blocks colored by split (clipped to the state) and the outline."""
    image, extent = read_composite_rgb(src, ags_limits)
    ax.imshow(image, extent=extent)
    state = gpd.read_file(STATE_GPKG, layer="state").to_crs(src.crs)
    blocks = gpd.clip(block_polygons(src.crs), state)  # cells at the edge stick out of the state
    for split_name, color in SPLIT_COLORS.items():
        subset = blocks[blocks["split"] == split_name]
        subset.plot(ax=ax, facecolor=color, edgecolor=color, alpha=BLOCK_ALPHA, linewidth=0.3)
    state.boundary.plot(ax=ax, color="#222222", linewidth=0.6)
    handles = []
    for split_name, color in SPLIT_COLORS.items():
        label = f"{SPLIT_NAMES[split_name]} blocks"
        handles.append(Patch(facecolor=color, edgecolor=color, alpha=BLOCK_ALPHA, label=label))
    ax.legend(handles=handles, loc="upper left", fontsize=MIN_FONT, frameon=False)
    ax.set_axis_off()


def example_patches() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Raw (13, 64, 64) EuroSAT and Aguascalientes patches of EXAMPLE_CLASSES (same order)."""
    samples = pd.read_csv(SAMPLES_CSV, dtype={"cell_id": str})
    eurosat_test = EuroSATDataset("test", bands="ms13", cache_dir=ROOT / "data/eurosat/cache")
    eurosat, ags = [], []
    for name in EXAMPLE_CLASSES:
        row = samples[(samples["figure"] == "gradcam_eurosat") & (samples["true_label"] == name)]
        eurosat.append(eurosat_test[int(row["index"].iloc[0])]["image"].numpy())
        row = samples[(samples["figure"] == "gradcam_ags") & (samples["true_label"] == name)]
        with rasterio.open(PATCHES_DIR / f"{row['cell_id'].iloc[0]}.tif") as patch_src:
            ags.append(patch_src.read().astype(np.float32))
    return eurosat, ags


def write_study_area(path: Path) -> None:
    """Fig. 1: map of the split (a) and EuroSAT vs Aguascalientes examples (b)."""
    fig = plt.figure(figsize=(3.5, 5.4), layout="constrained")
    map_fig, patches_fig = fig.subfigures(2, 1, height_ratios=[3.0, 2.2])
    with rasterio.open(COMPOSITE) as src:
        stretch = rgb_stretch_limits(src)  # global 2-98 % of the composite, as the previews
        ags_limits = np.array([stretch[name] for name in RGB], dtype=np.float32)  # (3, 2)
        draw_map(map_fig.add_subplot(), src, ags_limits)
    map_fig.suptitle("(a)", fontsize=MIN_FONT)

    eurosat, ags = example_patches()
    eurosat_limits = rgb_limits(np.stack(eurosat))
    axes = patches_fig.subplots(2, 3)
    for col, name in enumerate(EXAMPLE_CLASSES):
        axes[0, col].imshow(to_rgb_image(eurosat[col], eurosat_limits))
        axes[0, col].set_title(name.replace("Herbaceous", "Herbaceous\n"), fontsize=MIN_FONT)
        axes[1, col].imshow(to_rgb_image(ags[col], ags_limits))
    axes[0, 0].set_ylabel("EuroSAT", fontsize=MIN_FONT)
    axes[1, 0].set_ylabel("Aguascalientes", fontsize=MIN_FONT)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    patches_fig.suptitle("(b)", fontsize=MIN_FONT)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    write_config_table(cfg, TABLES_DIR / "config.tex")
    write_classes_table(load_labels(LABELS_CSV), TABLES_DIR / "ags_classes.tex")
    write_study_area(FIGURES_DIR / "study_area.pdf")


if __name__ == "__main__":
    main()
