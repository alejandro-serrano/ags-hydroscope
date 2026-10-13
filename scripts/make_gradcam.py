"""Make the three Grad-CAM figures of the paper.

Run: python scripts/make_gradcam.py
Outputs in paper/figures/: gradcam_eurosat.pdf, gradcam_ags.pdf, gradcam_selected.pdf and
gradcam_samples.csv (which patches are shown, and the rule that chose each one).
The figures contain no numbers: only the predicted class and whether it is correct.

Selection rules (seed 0):
- gradcam_eurosat: one random EuroSAT test patch per class of SIX_CLASSES.
- gradcam_ags: one random Aguascalientes test patch per class of SIX_CLASSES.
- gradcam_selected: the same patches as gradcam_ags, except that the HerbaceousVegetation and
  SeaLake patches are replaced by 2 random HerbaceousVegetation test patches that the
  last_block model predicts as Forest (both placed in the HerbaceousVegetation row).

RGB stretch: 2-98 % per band, computed once per dataset. One stretch from the EuroSAT patches
shown, another from all the Aguascalientes patches shown (gradcam_ags and gradcam_selected).
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: we only save PDFs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hydroscope.data.ags import AguascalientesDataset
from hydroscope.data.eurosat import CLASSES, EuroSATDataset
from hydroscope.explain.gradcam import (
    compute_gradcam,
    overlay_cam,
    prepare_patch,
    rgb_limits,
    to_rgb_image,
)
from hydroscope.training.evaluate import load_checkpoint, read_meta
from hydroscope.training.train import set_seed

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "paper" / "figures"

MODEL_NAMES = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "vit_small": "ViT-S"}
MIN_FONT = 8
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts in the PDF
plt.rcParams["font.size"] = MIN_FONT

SEED = 0
DEVICE = torch.device("cpu")  # CPU: same maps on every run

SIX_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Industrial",
    "Residential",
    "SeaLake",
]
RGB_CKPTS = ["resnet50_rgb_imagenet", "efficientnet_b0_rgb_imagenet", "vit_small_rgb_imagenet"]
SSL4EO_ZERO_SHOT = "resnet50_ms13_ssl4eo"
SSL4EO_LAST_BLOCK = "resnet50_ms13_ssl4eo_ft-last_block"

RULE_ONE_PER_CLASS = "seed 0, one random test patch per class"
RULE_SAME_AS_AGS = "same patch as gradcam_ags"
RULE_HV_TO_FOREST = "seed 0, random HerbaceousVegetation test patch predicted Forest by last_block"


def load_model(run_id: str) -> tuple[torch.nn.Module, dict, str, torch.Tensor, torch.Tensor]:
    """Load a checkpoint. Returns (model, meta, model name, mean, std)."""
    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    meta = read_meta(ckpt_path)
    model = load_checkpoint(ckpt_path, meta, DEVICE)
    mean = torch.tensor(meta["normalization"]["mean"])
    std = torch.tensor(meta["normalization"]["std"])
    return model, meta, meta["model"], mean, std


def indices_of_class(labels: list[int], class_name: str) -> list[int]:
    """Positions in ``labels`` whose class is ``class_name``."""
    class_id = CLASSES.index(class_name)
    return [i for i, label in enumerate(labels) if label == class_id]


def pick_one_per_class(labels: list[int], class_names: list[str]) -> list[int]:
    """Seeded random choice of one patch for each class name."""
    set_seed(SEED)
    chosen = []
    for class_name in class_names:
        chosen.append(random.choice(indices_of_class(labels, class_name)))
    return chosen


def raw_patch_ags(ags_test: AguascalientesDataset, index: int) -> np.ndarray:
    """Raw uint16 patch of the Aguascalientes test set as float32 (13, 64, 64)."""
    return ags_test.patches[index].astype(np.float32)


def predict_ags_class(run_id: str, ags_test: AguascalientesDataset, class_name: str) -> list[int]:
    """Predicted class index of every test patch whose true class is ``class_name``."""
    model, meta, _, mean, std = load_model(run_id)
    preds = []
    with torch.no_grad():
        for i in indices_of_class(ags_test.labels, class_name):
            raw = torch.from_numpy(raw_patch_ags(ags_test, i))
            x = prepare_patch(raw, meta["bands"], mean, std).unsqueeze(0)  # (1, C, 64, 64)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
            preds.append(int(model(x).argmax(dim=1).item()))
    return preds


def pick_herbaceous_to_forest(ags_test: AguascalientesDataset) -> list[int]:
    """Two seeded-random HerbaceousVegetation test patches that last_block predicts as Forest."""
    hv_indices = indices_of_class(ags_test.labels, "HerbaceousVegetation")
    preds = predict_ags_class(SSL4EO_LAST_BLOCK, ags_test, "HerbaceousVegetation")
    wrong = []
    for i, pred in zip(hv_indices, preds, strict=True):
        if pred == CLASSES.index("Forest"):
            wrong.append(i)

    # Sanity check against the confusion matrix of the result JSON (read only).
    path = RESULTS_DIR / f"{SSL4EO_LAST_BLOCK}_ags_test.json"
    confusion = json.loads(path.read_text())["confusion"]
    expected = confusion[CLASSES.index("HerbaceousVegetation")][CLASSES.index("Forest")]
    if len(wrong) != expected:
        raise ValueError(f"found {len(wrong)} HV->Forest patches, confusion matrix says {expected}")
    print(f"HerbaceousVegetation predicted as Forest by last_block: {len(wrong)} patches")

    set_seed(SEED)
    return random.sample(wrong, 2)


def selected_patches(ags_ids: list[int], hv_wrong_ids: list[int]) -> list[tuple[int, str]]:
    """Rows of gradcam_selected as (index, selection rule).

    The gradcam_ags patches (one per class of SIX_CLASSES, same order) are reused; the
    HerbaceousVegetation patch is replaced by the 2 HerbaceousVegetation->Forest cases and the
    SeaLake patch is dropped, so the figure keeps 6 rows.
    """
    rows = []
    for class_name, i in zip(SIX_CLASSES, ags_ids, strict=True):
        if class_name == "HerbaceousVegetation":
            for j in hv_wrong_ids:
                rows.append((j, RULE_HV_TO_FOREST))
        elif class_name != "SeaLake":
            rows.append((i, RULE_SAME_AS_AGS))
    return rows


def draw_figure(
    path: Path, rows: list[dict], columns: list[tuple[str, str]], limits: np.ndarray
) -> None:
    """Draw a grid: one row per patch, first column RGB, then one Grad-CAM column per model.

    ``rows`` has the keys raw (13, 64, 64) and true_label; ``columns`` is a list of
    (title, run_id).
    """
    n_rows = len(rows)
    n_cols = len(columns) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 1.95 * n_rows))

    # Every model is loaded once and used for all rows.
    for col, (title, run_id) in enumerate(columns, start=1):
        model, meta, name, mean, std = load_model(run_id)
        axes[0, col].set_title(title, fontsize=MIN_FONT)
        for row, item in enumerate(rows):
            raw = torch.from_numpy(item["raw"])
            image = prepare_patch(raw, meta["bands"], mean, std)  # (C, 64, 64)
            cam, pred = compute_gradcam(model, name, image)
            rgb = to_rgb_image(item["raw"], limits)
            ax = axes[row, col]
            ax.imshow(overlay_cam(rgb, cam))
            mark = "✓" if CLASSES[pred] == item["true_label"] else "✗"
            ax.set_xlabel(f"{CLASSES[pred]} {mark}", fontsize=MIN_FONT)

    axes[0, 0].set_title("RGB", fontsize=MIN_FONT)
    for row, item in enumerate(rows):
        axes[row, 0].imshow(to_rgb_image(item["raw"], limits))
        axes[row, 0].set_ylabel(item["true_label"], fontsize=MIN_FONT)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eurosat_test = EuroSATDataset("test", bands="ms13", cache_dir="data/eurosat/cache")
    eurosat_labels = [int(eurosat_test[i]["label"]) for i in range(len(eurosat_test))]
    ags_test = AguascalientesDataset("test", "ms13", torch.zeros(13), torch.ones(13), train=False)

    # 1. Choose the patches of the three figures (seed 0, rules in the module docstring).
    eurosat_ids = pick_one_per_class(eurosat_labels, SIX_CLASSES)
    ags_ids = pick_one_per_class(ags_test.labels, SIX_CLASSES)
    hv_wrong_ids = pick_herbaceous_to_forest(ags_test)
    selected = selected_patches(ags_ids, hv_wrong_ids)  # list of (index, rule)

    eurosat_rows = []
    for i in eurosat_ids:
        raw = eurosat_test[i]["image"].numpy()  # (13, 64, 64) raw DN
        eurosat_rows.append({"raw": raw, "true_label": CLASSES[eurosat_labels[i]]})
    ags_rows = []
    for i in ags_ids:
        ags_rows.append(
            {"raw": raw_patch_ags(ags_test, i), "true_label": CLASSES[ags_test.labels[i]]}
        )
    selected_rows = []
    for i, _ in selected:
        selected_rows.append(
            {"raw": raw_patch_ags(ags_test, i), "true_label": CLASSES[ags_test.labels[i]]}
        )

    # 2. One RGB stretch (2-98 %) per dataset. Aguascalientes: every patch shown, each once.
    eurosat_limits = rgb_limits(np.stack([r["raw"] for r in eurosat_rows]))
    ags_shown = list(ags_ids)
    for i in hv_wrong_ids:
        if i not in ags_shown:
            ags_shown.append(i)
    ags_limits = rgb_limits(np.stack([raw_patch_ags(ags_test, i) for i in ags_shown]))

    # 3. Save the list of patches.
    table = []
    for i in eurosat_ids:
        label = CLASSES[eurosat_labels[i]]
        table.append(["gradcam_eurosat", "eurosat_test", i, "", label, RULE_ONE_PER_CLASS])
    for i in ags_ids:
        cell_id = ags_test.cell_ids[i]
        label = CLASSES[ags_test.labels[i]]
        table.append(["gradcam_ags", "ags_test", i, cell_id, label, RULE_ONE_PER_CLASS])
    for i, rule in selected:
        cell_id = ags_test.cell_ids[i]
        label = CLASSES[ags_test.labels[i]]
        table.append(["gradcam_selected", "ags_test", i, cell_id, label, rule])
    columns = ["figure", "dataset", "index", "cell_id", "true_label", "selection_rule"]
    pd.DataFrame(table, columns=columns).to_csv(FIGURES_DIR / "gradcam_samples.csv", index=False)

    # 4. Draw.
    rgb_columns = []
    for run_id in RGB_CKPTS:
        rgb_columns.append((MODEL_NAMES[run_id.rsplit("_rgb", 1)[0]], run_id))
    draw_figure(FIGURES_DIR / "gradcam_eurosat.pdf", eurosat_rows, rgb_columns, eurosat_limits)
    draw_figure(FIGURES_DIR / "gradcam_ags.pdf", ags_rows, rgb_columns, ags_limits)
    selected_columns = [
        ("ResNet-50 MS13 SSL4EO\nzero-shot", SSL4EO_ZERO_SHOT),
        ("ResNet-50 MS13 SSL4EO\nlast_block", SSL4EO_LAST_BLOCK),
    ]
    draw_figure(FIGURES_DIR / "gradcam_selected.pdf", selected_rows, selected_columns, ags_limits)


if __name__ == "__main__":
    main()
