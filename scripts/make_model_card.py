"""Write docs/model_card.md from paper/tables/summary.csv.

Run: python scripts/make_model_card.py   (after python scripts/make_results.py)

Every metric in the card comes from paper/tables/summary.csv; the selection uses the frozen rule
of docs/protocol.md section 8 (``select_model`` of scripts/make_results.py). Patch counts come
from data/ags/labels.csv and the EuroSAT cache. Edit this script, never the card by hand.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from make_results import MODEL_NAMES, TIE_MARGIN, fmt3, select_model

from hydroscope.data.eurosat import CLASSES
from hydroscope.data.labels import SKIP, load_labels
from hydroscope.training.evaluate import MIN_TEST_PATCHES

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "paper" / "tables" / "summary.csv"
LABELS_CSV = ROOT / "data" / "ags" / "labels.csv"
EUROSAT_CACHE = ROOT / "data" / "eurosat" / "cache"
CARD_PATH = ROOT / "docs" / "model_card.md"

INIT_TEXT = {
    "imagenet": "ImageNet (timm)",
    "ssl4eo": "SSL4EO-S12 self-supervised Sentinel-2 weights (TorchGeo)",
}
BANDS_TEXT = {"rgb": "3 bands (B04, B03, B02)", "ms13": "13 Sentinel-2 bands (EuroSAT order)"}


def eurosat_sizes() -> dict[str, int]:
    """Number of EuroSAT patches per split, read from the .npy label cache."""
    sizes = {}
    for split in ["train", "val", "test"]:
        sizes[split] = len(np.load(EUROSAT_CACHE / f"{split}_y.npy", mmap_mode="r"))
    return sizes


def parse_cell_id(cell_id: str) -> tuple[int, int]:
    """'r0083_c0148' -> (83, 148)."""
    row_part, col_part = cell_id.split("_")
    return int(row_part[1:]), int(col_part[1:])


def count_test_next_to_finetune(labels: pd.DataFrame) -> int:
    """Labeled test cells with at least one labeled fine-tune cell among their 8 neighbours."""
    finetune_cells = set()
    for cell_id in labels[labels["split"] == "finetune"]["cell_id"]:
        finetune_cells.add(parse_cell_id(cell_id))
    count = 0
    for cell_id in labels[labels["split"] == "test"]["cell_id"]:
        row, col = parse_cell_id(cell_id)
        touches = False
        for d_row in [-1, 0, 1]:
            for d_col in [-1, 0, 1]:
                if (row + d_row, col + d_col) in finetune_cells:
                    touches = True
        if touches:
            count += 1
    return count


def ags_facts() -> dict:
    """Counts of the Aguascalientes splits used in the card (skip rows left out)."""
    labels = load_labels(LABELS_CSV)
    labels = labels[labels["label"] != SKIP]
    test = labels[labels["split"] == "test"]
    test_counts = test["label"].value_counts()
    macro_classes = []
    excluded = []
    for name in CLASSES:
        if test_counts.get(name, 0) >= MIN_TEST_PATCHES:
            macro_classes.append(name)
        else:
            excluded.append(f"{name} ({test_counts.get(name, 0)} test patches)")
    return {
        "n_finetune": int((labels["split"] == "finetune").sum()),
        "n_test": len(test),
        "n_test_blocks": test["block_id"].nunique(),
        "macro_classes": macro_classes,
        "excluded": excluded,
        "n_next_to_finetune": count_test_next_to_finetune(labels),
    }


def describe_run(row: pd.Series) -> str:
    """'ResNet-50, 13 Sentinel-2 bands (EuroSAT order), SSL4EO-S12 ... weights'."""
    return f"{MODEL_NAMES[row['model']]}, {BANDS_TEXT[row['bands']]}, {INIT_TEXT[row['init']]}"


def model_section(selected: pd.Series, secondary: pd.Series, selection: dict) -> list[str]:
    """Model details and how the model was chosen."""
    if selection["tie_break_applied"]:
        tie_text = "the tie-break on patches/s was applied"
    else:
        tie_text = f"no other run was within {TIE_MARGIN}, so no tie-break was needed"
    return [
        "## Model details",
        "",
        f"- **Selected model:** {selected['label']} with `last_block` fine-tuning",
        f"  (checkpoint `checkpoints/{selected['run_id']}_ft-last_block.pt`, GitHub Releases).",
        f"- Architecture and input: {describe_run(selected)}; 64x64 patches resized to 224x224.",
        "- Training: EuroSAT training with the shared protocol (`docs/protocol.md`), then",
        "  `last_block` fine-tuning on the Aguascalientes fine-tune split (last stage + head).",
        "- **Selection rule** (`docs/protocol.md` section 8, frozen): highest Aguascalientes",
        f"  macro-F1 after `last_block`; runs within {TIE_MARGIN} are decided by patches/s.",
        f"  Among the 8 runs, the margin over the runner-up is {fmt3(selection['margin'])} and",
        f"  {tie_text}.",
        f"- **Secondary result:** {secondary['label']}, the best of the 6 ImageNet runs by the",
        "  same rule: the best observed result without self-supervised Sentinel-2 weights. The",
        "  gap between the two cannot be credited to the weights or to the architecture alone:",
        "  SSL4EO runs use a different normalization (x / 10000), and `last_block` trains a",
        "  different share of the parameters of each architecture (`docs/protocol.md` section 4).",
        "",
    ]


def use_and_data_section(facts: dict, sizes: dict[str, int]) -> list[str]:
    """Intended use, training data and evaluation data."""
    excluded = ", ".join(facts["excluded"])
    return [
        "## Intended use",
        "",
        "- Land-use classification of the 640 m grid cells of Aguascalientes, Mexico, into the 10",
        "  EuroSAT classes, from a Sentinel-2 L1C median composite of April 2024.",
        "- Base model for stage 2 (monthly 2017-2026 maps and municipal indicators of the web app",
        "  Hydroscope Aguascalientes). Monthly maps are outside what was evaluated here.",
        "- Not intended for other regions or dates without new validation, for pixel-level",
        "  mapping, or for decisions about individual parcels.",
        "",
        "## Training data",
        "",
        f"- EuroSAT MS (TorchGeo official split): {sizes['train']} train, {sizes['val']} val and",
        f"  {sizes['test']} test patches, 13 x 64 x 64 at 10 m.",
        f"- Aguascalientes fine-tune split: {facts['n_finetune']} labeled patches",
        "  (`docs/data_card.md`), class-weighted cross-entropy, 10 fixed epochs, no validation.",
        "",
        "## Evaluation data",
        "",
        f"- Aguascalientes test split: {facts['n_test']} labeled patches from",
        f"  {facts['n_test_blocks']} spatial blocks of 8x8 cells (5.12 km) that hold at least one",
        "  labeled patch; no block is shared with the fine-tune split.",
        f"- Macro-F1 averages the {len(facts['macro_classes'])} classes with at least",
        f"  {MIN_TEST_PATCHES} test patches. Left out of it and reported apart: {excluded}.",
        "  The macro-F1 over all 10 classes is also given below.",
        f"- EuroSAT test split ({sizes['test']} patches) for the source-domain results.",
        "",
    ]


def metrics_section(selected: pd.Series, secondary: pd.Series, n_macro: int) -> list[str]:
    """Metrics table, selected vs secondary model."""
    rows = [
        ("EuroSAT test accuracy", "eurosat_accuracy"),
        ("EuroSAT test macro-F1", "eurosat_macro_f1"),
        (f"Aguascalientes macro-F1, zero-shot ({n_macro} classes)", "ags_zero_shot"),
        (f"Aguascalientes macro-F1, `head` ({n_macro} classes)", "ags_head"),
        (f"Aguascalientes macro-F1, `last_block` ({n_macro} classes)", "ags_last_block"),
        ("Aguascalientes macro-F1, `last_block` (10 classes)", "ags_last_block_all_classes"),
        ("Drop (EuroSAT -> Aguascalientes zero-shot)", "drop"),
        ("Recovery with `last_block`", "recovery_last_block"),
    ]
    lines = ["## Metrics", "", f"| Metric | {selected['label']} | {secondary['label']} |"]
    lines.append("| --- | ---: | ---: |")
    for name, column in rows:
        lines.append(f"| {name} | {fmt3(selected[column])} | {fmt3(secondary[column])} |")
    lines.append("")
    lines.append("Drop and Recovery follow `docs/protocol.md` section 7. Per-class F1 and the")
    lines.append("confusion matrix are in `paper/figures/` (`ags_f1_per_class.pdf`,")
    lines.append("`confusion_selected.pdf`).")
    lines.append("")
    return lines


def cost_section(selected: pd.Series, secondary: pd.Series) -> list[str]:
    """Computational cost of the EuroSAT training run of both models."""
    rows = [
        ("Parameters", f"{int(selected['params']):,}", f"{int(secondary['params']):,}"),
        ("EuroSAT epochs", f"{int(selected['epochs'])}", f"{int(secondary['epochs'])}"),
        (
            "EuroSAT training time (min)",
            f"{selected['train_time_s'] / 60:.1f}",
            f"{secondary['train_time_s'] / 60:.1f}",
        ),
        (
            "Peak memory (MB)",
            f"{selected['peak_mem_mb']:.0f}",
            f"{secondary['peak_mem_mb']:.0f}",
        ),
        (
            "Inference (patches/s, EuroSAT test)",
            f"{selected['patches_per_s']:.1f}",
            f"{secondary['patches_per_s']:.1f}",
        ),
        (
            "Epochs to 95 % of best val macro-F1",
            f"{int(selected['epochs_to_95'])}",
            f"{int(secondary['epochs_to_95'])}",
        ),
    ]
    lines = ["## Cost", "", f"| | {selected['label']} | {secondary['label']} |"]
    lines.append("| --- | ---: | ---: |")
    for name, value_a, value_b in rows:
        lines.append(f"| {name} | {value_a} | {value_b} |")
    lines.append("")
    lines.append("Apple M2 Pro (MPS), fp32, effective batch 64 = 2 x 32 (`docs/protocol.md`).")
    lines.append("")
    return lines


def limitations_section(facts: dict) -> list[str]:
    """Known limitations of the selected model and of its evaluation."""
    return [
        "## Limitations",
        "",
        "- **One seed** (0) per run: there is no variance estimate, so small differences between",
        f"  runs (within {TIE_MARGIN} macro-F1) should not be read as real.",
        "- **One month:** a single April 2024 composite (dry season). Other months, the rainy",
        "  season and other years are not evaluated.",
        "- **One annotator** (`A1`): no inter-annotator agreement, label noise is unmeasured.",
        "- **Scrub / forest confusion:** after `last_block`, xeric scrub labeled",
        "  HerbaceousVegetation is often predicted as Forest (`confusion_selected.pdf`,",
        "  `gradcam_selected.pdf`).",
        "- **Objects without a EuroSAT class**, such as solar parks, are always forced into one of",
        "  the 10 classes. Cells of this kind (greenhouses, quarries, bare rock or soil) were",
        "  marked `skip` while labeling, so the test set does not measure this error.",
        f"- **Test cells next to fine-tune cells:** {facts['n_next_to_finetune']} of the",
        f"  {facts['n_test']} test patches have a fine-tune patch among their 8 neighbours (block",
        "  borders). Spatial autocorrelation can make the fine-tuned scores optimistic.",
        "- Rare classes have few test patches, so their F1 is noisy.",
        "",
    ]


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    all_runs = list(summary["run_id"])
    imagenet_runs = list(summary[summary["init"] == "imagenet"]["run_id"])
    selection = select_model(summary, all_runs)
    selection_imagenet = select_model(summary, imagenet_runs)
    by_id = summary.set_index("run_id", drop=False)
    selected = by_id.loc[selection["winner"]]
    secondary = by_id.loc[selection_imagenet["winner"]]
    facts = ags_facts()

    lines = [
        "# Model card: Aguascalientes land-use classifier",
        "",
        "<!-- Generated by scripts/make_model_card.py from paper/tables/summary.csv."
        " Do not edit by hand. -->",
        "",
    ]
    lines += model_section(selected, secondary, selection)
    lines += use_and_data_section(facts, eurosat_sizes())
    lines += metrics_section(selected, secondary, len(facts["macro_classes"]))
    lines += cost_section(selected, secondary)
    lines += limitations_section(facts)
    CARD_PATH.write_text("\n".join(lines))
    print(f"selected: {selected['label']} | secondary: {secondary['label']}")
    print(f"wrote {CARD_PATH}")


if __name__ == "__main__":
    main()
