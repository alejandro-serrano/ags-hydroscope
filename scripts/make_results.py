"""Make the paper tables and figures from experiments/results/*.json and experiments/logs/*.csv.

Run: python scripts/make_results.py
Every number comes from the JSON results or the CSV training logs; nothing is typed by hand.
Outputs: paper/tables/*.tex, paper/tables/summary.csv, paper/tables/macros.tex (the numbers
quoted in the text, read from summary.csv) and paper/figures/*.pdf.
macros.tex also holds two numbers that are not results: the per-band z-scores of the
Aguascalientes composite, read from docs/data_ags_check.md (written by
scripts/check_ags_raster.py), the share of parameters trained in last_block mode, counted
with the model factory, and two fine-tuning numbers: the optimizer steps of one run (from
configs/base.yaml and data/ags/labels.csv) and how many head runs still lowered their training
loss in the second half (from experiments/logs/*_ft-head.csv).
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: we only save PDFs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from hydroscope.config import load_config
from hydroscope.data.eurosat import BAND_SETS
from hydroscope.data.labels import SKIP, load_labels
from hydroscope.models.factory import build_model, count_params, last_block_params

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
LOGS_DIR = ROOT / "experiments" / "logs"
TABLES_DIR = ROOT / "paper" / "tables"
FIGURES_DIR = ROOT / "paper" / "figures"
AGS_CHECK_REPORT = ROOT / "docs" / "data_ags_check.md"
BASE_CONFIG = ROOT / "configs" / "base.yaml"
AGS_LABELS = ROOT / "data" / "ags" / "labels.csv"

MODEL_ORDER = ["resnet50", "efficientnet_b0", "vit_small"]
BANDS_ORDER = ["rgb", "ms13"]
INIT_ORDER = ["imagenet", "ssl4eo"]

MODEL_NAMES = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "vit_small": "ViT-S"}
BANDS_NAMES = {"rgb": "RGB", "ms13": "MS13"}
INIT_NAMES = {"imagenet": "ImageNet", "ssl4eo": "SSL4EO"}

# One color per model, fixed order (validated with the dataviz palette checker).
MODEL_COLORS = {"resnet50": "#2a78d6", "efficientnet_b0": "#eb6834", "vit_small": "#1baf7a"}
INIT_LINESTYLES = {"imagenet": "-", "ssl4eo": "--"}

INK = "#222222"  # text and axes
GRAY = "#777777"
BLUE_LIGHT = "#9ecae1"  # Blues ramp, light and dark step
BLUE_DARK = "#08519c"

TIE_MARGIN = 0.01  # protocol section 8: within 1 point of macro-F1
MIN_FONT = 8


# ---------------------------------------------------------------- loading


def run_label(model: str, bands: str, init: str) -> str:
    """Display label, e.g. 'ResNet-50 / RGB / ImageNet'."""
    return f"{MODEL_NAMES[model]} / {BANDS_NAMES[bands]} / {INIT_NAMES[init]}"


def load_results(results_dir: Path) -> dict[str, dict]:
    """Read the 4 result JSONs of every run. Returns {run_id: {'eurosat': ..., 'ags': ...}}."""
    run_ids = []
    for path in results_dir.glob("*_eurosat_test.json"):
        run_ids.append(path.name.removesuffix("_eurosat_test.json"))

    suffixes = {
        "eurosat": "_eurosat_test",
        "ags": "_ags_test",
        "head": "_ft-head_ags_test",
        "last_block": "_ft-last_block_ags_test",
    }
    results = {}
    for run_id in run_ids:
        record = {}
        for key, suffix in suffixes.items():
            path = results_dir / f"{run_id}{suffix}.json"
            if not path.exists():
                raise ValueError(f"missing result file: {path.name}")
            with open(path) as f:
                record[key] = json.load(f)
        results[run_id] = record

    def sort_key(run_id: str) -> tuple[int, int, int]:
        r = results[run_id]["eurosat"]
        return (
            MODEL_ORDER.index(r["model"]),
            BANDS_ORDER.index(r["bands"]),
            INIT_ORDER.index(r["init"]),
        )

    ordered = sorted(results, key=sort_key)
    return {run_id: results[run_id] for run_id in ordered}


def macro_classes(results: dict[str, dict]) -> list[str]:
    """The class list used by the Aguascalientes macro-F1. It must be the same in every JSON."""
    class_lists = []
    for record in results.values():
        for key in ["ags", "head", "last_block"]:
            class_lists.append(record[key]["macro_f1_classes"])
    first = class_lists[0]
    for other in class_lists:
        if other != first:
            raise ValueError("macro_f1_classes differs between result files")
    return first


# ---------------------------------------------------------------- metrics


def eurosat_f1_on(eurosat_record: dict, class_names: list[str]) -> float:
    """Mean of the per-class F1 of the EuroSAT test result over the given classes."""
    values = [eurosat_record["f1_per_class"][name] for name in class_names]
    return sum(values) / len(values)


def drop(f1_eurosat9: float, f1_zero_shot: float) -> float:
    """Protocol section 7: Drop = F1 EuroSAT - F1 Aguascalientes zero-shot."""
    return f1_eurosat9 - f1_zero_shot


def recovery(f1_ft: float, f1_zero_shot: float, drop_value: float) -> float:
    """Protocol section 7: Recovery = (F1 fine-tuned - F1 zero-shot) / Drop. NaN if Drop <= 0."""
    if drop_value <= 0:
        return float("nan")
    return (f1_ft - f1_zero_shot) / drop_value


def count_log_rows(logs_dir: Path, run_id: str) -> int:
    """Number of epochs actually trained = rows of the training log."""
    return len(pd.read_csv(logs_dir / f"{run_id}.csv"))


def build_summary(results: dict[str, dict], logs_dir: Path) -> pd.DataFrame:
    """One row per run with every number the tables and figures need."""
    classes = macro_classes(results)
    rows = []
    for run_id, record in results.items():
        eurosat = record["eurosat"]
        f1_eurosat9 = eurosat_f1_on(eurosat, classes)
        zero_shot = record["ags"]["macro_f1"]
        head = record["head"]["macro_f1"]
        last_block = record["last_block"]["macro_f1"]
        drop_value = drop(f1_eurosat9, zero_shot)
        epochs = count_log_rows(logs_dir, run_id)
        rows.append(
            {
                "run_id": run_id,
                "label": run_label(eurosat["model"], eurosat["bands"], eurosat["init"]),
                "model": eurosat["model"],
                "bands": eurosat["bands"],
                "init": eurosat["init"],
                "eurosat_accuracy": eurosat["accuracy"],
                "eurosat_macro_f1": eurosat["macro_f1"],
                "eurosat_f1_macro_classes": f1_eurosat9,
                "ags_zero_shot": zero_shot,
                "ags_head": head,
                "ags_last_block": last_block,
                "ags_zero_shot_all_classes": record["ags"]["macro_f1_all_classes"],
                "ags_last_block_all_classes": record["last_block"]["macro_f1_all_classes"],
                "ags_zero_shot_accuracy": record["ags"]["accuracy"],
                "ags_head_accuracy": record["head"]["accuracy"],
                "ags_last_block_accuracy": record["last_block"]["accuracy"],
                "drop": drop_value,
                "recovery_head": recovery(head, zero_shot, drop_value),
                "recovery_last_block": recovery(last_block, zero_shot, drop_value),
                "params": eurosat["params"],
                "epochs": epochs,
                "train_time_s": eurosat["train_time_s"],
                "time_per_epoch_s": eurosat["train_time_s"] / epochs,
                "peak_mem_mb": eurosat["peak_mem_mb"],
                "patches_per_s": eurosat["patches_per_s"],
                "epochs_to_95": eurosat["epochs_to_95"],
            }
        )
    return pd.DataFrame(rows)


def select_model(summary: pd.DataFrame, run_ids: list[str]) -> dict:
    """Protocol section 8: best last_block macro-F1; ties within 0.01 go to higher patches/s."""
    subset = summary[summary["run_id"].isin(run_ids)]
    by_f1 = subset.sort_values("ags_last_block", ascending=False)
    top = by_f1.iloc[0]
    runner_up = by_f1.iloc[1]

    close = subset[top["ags_last_block"] - subset["ags_last_block"] <= TIE_MARGIN]
    winner = close.sort_values("patches_per_s", ascending=False).iloc[0]
    tie_break = len(close) > 1

    result = {
        "winner": winner["run_id"],
        "top_f1_run": top["run_id"],
        "runner_up": runner_up["run_id"],
        "margin": top["ags_last_block"] - runner_up["ags_last_block"],
        "tie_break_applied": tie_break,
        "candidates": list(close["run_id"]),
        "throughput_difference": None,
    }
    if tie_break:
        result["throughput_difference"] = winner["patches_per_s"] - top["patches_per_s"]
    return result


def format_selection(name: str, selection: dict, summary: pd.DataFrame) -> str:
    """Text lines describing one selection result."""
    by_id = summary.set_index("run_id")

    def describe(run_id: str) -> str:
        row = by_id.loc[run_id]
        return (
            f"{row['label']} (last_block macro-F1 {row['ags_last_block']:.4f}, "
            f"{row['patches_per_s']:.1f} patches/s)"
        )

    lines = [f"Selection among {name}:"]
    lines.append(f"  winner: {describe(selection['winner'])}")
    lines.append(f"  highest macro-F1: {describe(selection['top_f1_run'])}")
    lines.append(f"  runner-up by macro-F1: {describe(selection['runner_up'])}")
    lines.append(f"  margin top vs runner-up: {selection['margin']:.4f}")
    lines.append(f"  tie-break applied: {selection['tie_break_applied']}")
    if selection["tie_break_applied"]:
        labels = [by_id.loc[r, "label"] for r in selection["candidates"]]
        lines.append(f"  candidates within {TIE_MARGIN}: {', '.join(labels)}")
        diff = selection["throughput_difference"]
        lines.append(f"  patches/s of winner minus patches/s of top-F1 run: {diff:.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------- tables


def fmt3(value: float) -> str:
    """Three decimals, or '--' for NaN."""
    if np.isnan(value):
        return "--"
    return f"{value:.3f}"


def eurosat_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Strings of Table II: EuroSAT test."""
    table = pd.DataFrame()
    table["Run"] = summary["label"]
    table["Accuracy"] = [fmt3(v) for v in summary["eurosat_accuracy"]]
    table["Macro-F1"] = [fmt3(v) for v in summary["eurosat_macro_f1"]]
    return table


def ags_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Strings of Table III: Aguascalientes test."""
    table = pd.DataFrame()
    table["Run"] = summary["label"]
    table["EuroSAT F1"] = [fmt3(v) for v in summary["eurosat_f1_macro_classes"]]
    table["Zero-shot"] = [fmt3(v) for v in summary["ags_zero_shot"]]
    table["Head"] = [fmt3(v) for v in summary["ags_head"]]
    table["Last block"] = [fmt3(v) for v in summary["ags_last_block"]]
    table["Drop"] = [fmt3(v) for v in summary["drop"]]
    table["Rec. head"] = [fmt3(v) for v in summary["recovery_head"]]
    table["Rec. last block"] = [fmt3(v) for v in summary["recovery_last_block"]]
    return table


def cost_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Strings of Table IV: cost."""
    table = pd.DataFrame()
    table["Run"] = summary["label"]
    table["Params (M)"] = [f"{v / 1e6:.1f}" for v in summary["params"]]
    table["Train time (min)"] = [f"{v / 60:.1f}" for v in summary["train_time_s"]]
    table["Time/epoch (s)"] = [f"{v:.0f}" for v in summary["time_per_epoch_s"]]
    table["Peak mem. (GiB)"] = [f"{v / 1024:.2f}" for v in summary["peak_mem_mb"]]
    table["Patches/s"] = [f"{v:.0f}" for v in summary["patches_per_s"]]
    return table


def latex_table(
    table: pd.DataFrame,
    caption: str,
    label: str,
    environment: str,
    bold_rows: dict[str, int],
) -> str:
    """Turn a table of strings into a booktabs float. bold_rows maps a column to the row to bold."""
    columns = list(table.columns)
    spec = "l" + "r" * (len(columns) - 1)
    lines = [f"\\begin{{{environment}}}[t]", "\\centering", "\\footnotesize"]
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\midrule")
    for i in range(len(table)):
        cells = []
        for column in columns:
            text = table[column].iloc[i]
            if column in bold_rows and bold_rows[column] == i:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines) + "\n"


def best_row(values: pd.Series) -> int:
    """Position of the largest value."""
    return int(np.argmax(values.to_numpy()))


def write_tables(summary: pd.DataFrame, classes: list[str], n_classes_total: int) -> list[Path]:
    """Write eurosat.tex, ags.tex, cost.tex and summary.csv."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    eurosat_caption = (
        "EuroSAT test set (10 classes): overall accuracy and macro-F1 of the eight runs. "
        "Best value per column in bold."
    )
    eurosat_bold = {
        "Accuracy": best_row(summary["eurosat_accuracy"]),
        "Macro-F1": best_row(summary["eurosat_macro_f1"]),
    }
    text = latex_table(
        eurosat_table(summary), eurosat_caption, "tab:eurosat", "table", eurosat_bold
    )
    path = TABLES_DIR / "eurosat.tex"
    path.write_text(text)
    written.append(path)

    excluded = n_classes_total - len(classes)
    ags_caption = (
        f"Macro-F1 on the Aguascalientes test split. The class set is the {len(classes)} classes "
        f"with at least 10 test patches ({excluded} class excluded and reported separately). "
        "EuroSAT F1 is the macro-F1 of the EuroSAT test result over the same classes. "
        "Drop = EuroSAT F1 minus zero-shot; recovery = (fine-tuned minus zero-shot) / drop, "
        "shown as -- when the drop is not positive. Best F1 per column in bold."
    )
    ags_bold = {
        "EuroSAT F1": best_row(summary["eurosat_f1_macro_classes"]),
        "Zero-shot": best_row(summary["ags_zero_shot"]),
        "Head": best_row(summary["ags_head"]),
        "Last block": best_row(summary["ags_last_block"]),
    }
    text = latex_table(ags_table(summary), ags_caption, "tab:ags", "table*", ags_bold)
    path = TABLES_DIR / "ags.tex"
    path.write_text(text)
    written.append(path)

    cost_caption = (
        "Training cost of the eight EuroSAT runs on an Apple M2 Pro (fp32). Peak memory is the "
        "memory reserved by the MPS driver. Patches/s is inference throughput on the EuroSAT "
        "test set."
    )
    text = latex_table(cost_table(summary), cost_caption, "tab:cost", "table*", {})
    path = TABLES_DIR / "cost.tex"
    path.write_text(text)
    written.append(path)

    path = TABLES_DIR / "summary.csv"
    summary.to_csv(path, index=False)
    written.append(path)
    return written


# ---------------------------------------------------------------- macros


def paper_macros(summary: pd.DataFrame, selection_all: dict, selection_imagenet: dict) -> dict:
    """Every number quoted in the paper text, as {macro name: text}. Same rounding as the tables."""
    by_id = summary.set_index("run_id")
    resnet_rgb = by_id.loc["resnet50_rgb_imagenet"]
    cnn = summary[summary["model"] != "vit_small"]
    vit = summary[summary["model"] == "vit_small"]
    accuracy_falls = summary["ags_last_block_accuracy"] < summary["ags_zero_shot_accuracy"]

    macros = {}
    # EuroSAT
    macros["EuroSATMin"] = fmt3(summary["eurosat_macro_f1"].min())
    macros["EuroSATMax"] = fmt3(summary["eurosat_macro_f1"].max())
    macros["EpochsToNinetyFiveMax"] = str(summary["epochs_to_95"].max())
    # Aguascalientes zero-shot
    macros["ZeroShotMin"] = fmt3(summary["ags_zero_shot"].min())
    macros["ZeroShotMax"] = fmt3(summary["ags_zero_shot"].max())
    macros["DropMin"] = fmt3(summary["drop"].min())
    macros["DropMax"] = fmt3(summary["drop"].max())
    # Fine-tuning
    macros["HeadRecoveryMax"] = fmt3(summary["recovery_head"].max())
    macros["RecoveryMin"] = fmt3(summary["recovery_last_block"].min())
    macros["RecoveryMax"] = fmt3(summary["recovery_last_block"].max())
    macros["AccuracyDropRuns"] = str(accuracy_falls.sum())
    macros["NumRuns"] = str(len(summary))
    # ResNet-50 / RGB / ImageNet
    macros["ResNetRgbHead"] = fmt3(resnet_rgb["ags_head"])
    macros["ResNetRgbLastBlock"] = fmt3(resnet_rgb["ags_last_block"])
    macros["ResNetRgbZeroShotAcc"] = fmt3(resnet_rgb["ags_zero_shot_accuracy"])
    macros["ResNetRgbLastBlockAcc"] = fmt3(resnet_rgb["ags_last_block_accuracy"])
    # Cost, same units as Table cost
    macros["PeakMemCnnMax"] = f"{cnn['peak_mem_mb'].max() / 1024:.2f}"
    macros["PeakMemVitMin"] = f"{vit['peak_mem_mb'].min() / 1024:.2f}"
    macros["PatchesPerSecMin"] = f"{summary['patches_per_s'].min():.0f}"
    macros["PatchesPerSecMax"] = f"{summary['patches_per_s'].max():.0f}"
    # Selection rule (protocol section 8)
    macros["AgsLastBlockBest"] = fmt3(by_id.loc[selection_all["winner"], "ags_last_block"])
    macros["SelectionMargin"] = fmt3(selection_all["margin"])
    macros["ImageNetLastBlockBest"] = fmt3(
        by_id.loc[selection_imagenet["winner"], "ags_last_block"]
    )
    macros["ImageNetSelectionMargin"] = fmt3(selection_imagenet["margin"])
    return macros


def band_zscores(report_path: Path) -> dict[str, float]:
    """z-score of each band, read from the band table of docs/data_ags_check.md.

    z = (Aguascalientes median - EuroSAT train mean) / EuroSAT train std.
    Table columns: Band, p2, median, p98, mean, std, z, Flag, Within 3 std.
    """
    zscores = {}
    for line in report_path.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 9 and cells[0] in BAND_SETS["ms13"]:
            zscores[cells[0]] = float(cells[6])
    if len(zscores) != 13:
        raise ValueError(f"expected 13 bands in {report_path}, found {len(zscores)}")
    return zscores


def band_macros(zscores: dict[str, float]) -> dict:
    """z-score macros for the Discussion: visible, NIR, SWIR and B10."""
    visible = [zscores["B02"], zscores["B03"], zscores["B04"]]
    nir = [zscores["B08"], zscores["B8A"]]
    swir = [zscores["B11"], zscores["B12"]]
    macros = {}
    macros["ZVisibleMin"] = f"{min(visible):.2f}"
    macros["ZVisibleMax"] = f"{max(visible):.2f}"
    macros["ZNirMax"] = f"{max(nir):.2f}"
    macros["ZSwirMin"] = f"{min(swir):.2f}"
    macros["ZSwirMax"] = f"{max(swir):.2f}"
    macros["ZBandTen"] = f"{zscores['B10']:.2f}"
    return macros


def last_block_shares() -> dict[str, float]:
    """Share of the parameters trained in last_block mode, per model (RGB, ImageNet weights)."""
    shares = {}
    for name in MODEL_ORDER:
        model = build_model(name, "rgb")
        trained = sum(p.numel() for p in last_block_params(model, name))
        shares[name] = trained / count_params(model)
    return shares


def share_macros(shares: dict[str, float]) -> dict:
    """last_block share macros, in percent without decimals."""
    macros = {}
    macros["LastBlockShareResNet"] = f"{100 * shares['resnet50']:.0f}"
    macros["LastBlockShareEffNet"] = f"{100 * shares['efficientnet_b0']:.0f}"
    macros["LastBlockShareViT"] = f"{100 * shares['vit_small']:.0f}"
    return macros


def finetune_steps(config_path: Path, labels_path: Path) -> int:
    """Optimizer steps of one fine-tuning run, counted as in hydroscope.training.finetune."""
    cfg = load_config(config_path)
    labels = load_labels(labels_path)
    is_finetune = (labels["split"] == "finetune") & (labels["label"] != SKIP)
    n_patches = int(is_finetune.sum())
    micro_batch = cfg["optim"]["micro_batch_size"]
    accum = cfg["optim"]["batch_size"] // micro_batch
    micro_batches = math.ceil(n_patches / micro_batch)  # len(train_loader)
    steps_per_epoch = math.ceil(micro_batches / accum)
    return cfg["finetune"]["epochs"] * steps_per_epoch


def head_loss_falling_runs(summary: pd.DataFrame, logs_dir: Path) -> int:
    """Number of head runs whose training loss at the last epoch is below that at the middle one."""
    count = 0
    for run_id in summary["run_id"]:
        log = pd.read_csv(logs_dir / f"{run_id}_ft-head.csv")
        middle = log.loc[log["epoch"] == len(log) // 2, "train_loss"].item()
        last = log["train_loss"].iloc[-1]
        if last < middle:
            count += 1
    return count


def write_macros(macros: dict) -> Path:
    """Write paper/tables/macros.tex, one \\newcommand per number."""
    lines = [
        "% Generated by scripts/make_results.py from paper/tables/summary.csv,",
        "% docs/data_ags_check.md (band z-scores), the model factory (last_block shares),",
        "% configs/base.yaml + data/ags/labels.csv (fine-tune steps) and the fine-tune logs.",
        "% Do not edit.",
    ]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    path = TABLES_DIR / "macros.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- figures


def style_axes(ax) -> None:
    """Ink-colored text and light grid."""
    ax.tick_params(colors=INK, labelsize=MIN_FONT)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    for spine in ax.spines.values():
        spine.set_color(GRAY)


def plot_drop_recovery(summary: pd.DataFrame) -> Path:
    """Fig. 2: (a) drop per run, (b) recovery of head and last_block."""
    n = len(summary)
    positions = np.arange(n)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)

    ax_a.barh(positions, summary["drop"], color=BLUE_DARK, height=0.6)
    ax_a.axvline(0, color=GRAY, linewidth=1)
    ax_a.set_title("(a) Drop (EuroSAT F1 - zero-shot F1)", fontsize=MIN_FONT + 1)
    ax_a.set_xlabel("Drop (macro-F1 points, 0-1)", fontsize=MIN_FONT)
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels(summary["label"], fontsize=MIN_FONT)
    ax_a.invert_yaxis()

    height = 0.35
    ax_b.barh(
        positions - height / 2,
        summary["recovery_head"],
        color=BLUE_LIGHT,
        height=height,
        label="head",
    )
    ax_b.barh(
        positions + height / 2,
        summary["recovery_last_block"],
        color=BLUE_DARK,
        height=height,
        label="last_block",
    )
    ax_b.axvline(0, color=GRAY, linewidth=1)
    ax_b.axvline(1, color=GRAY, linewidth=1, linestyle="--")
    ax_b.set_title("(b) Recovery (0 = none, 1 = full)", fontsize=MIN_FONT + 1)
    ax_b.set_xlabel("Recovery (fraction of the drop)", fontsize=MIN_FONT)
    ax_b.legend(fontsize=MIN_FONT, loc="lower right", frameon=True)

    for ax in [ax_a, ax_b]:
        style_axes(ax)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "drop_recovery.pdf"
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    return path


def class_names_from(results: dict[str, dict]) -> list[str]:
    """Class names in index order, from the first result file."""
    first = next(iter(results.values()))
    return list(first["ags"]["f1_per_class"].keys())


def draw_f1_heatmap(
    ax, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], hatched: list[int]
):
    """Blues heatmap 0-1 with the value in each cell. Hatched columns are not in the macro-F1."""
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > 0.55 else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=MIN_FONT, color=color)
    for j in hatched:
        # Hatch drawn on top, in gray, so the River column is clearly marked.
        rect = Rectangle(
            (j - 0.5, -0.5),
            1,
            matrix.shape[0],
            fill=False,
            hatch="//",
            edgecolor="#aaaaaa",
            linewidth=0,
        )
        ax.add_patch(rect)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=MIN_FONT)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=MIN_FONT)
    style_axes(ax)
    return image


def plot_f1_per_class(results: dict[str, dict], summary: pd.DataFrame) -> Path:
    """Per-class F1 on Aguascalientes: zero-shot (top) and last_block (bottom)."""
    names = class_names_from(results)
    macro = macro_classes(results)
    first = next(iter(results.values()))
    support = np.array(first["ags"]["confusion"]).sum(axis=1)  # (10,) test patches per class

    col_labels = []
    hatched = []
    for j, name in enumerate(names):
        if name in macro:
            col_labels.append(name)
        else:
            col_labels.append(f"{name} (reported separately, n={support[j]})")
            hatched.append(j)

    zero_shot = []
    last_block = []
    for record in results.values():
        zero_shot.append([record["ags"]["f1_per_class"][c] for c in names])
        last_block.append([record["last_block"]["f1_per_class"][c] for c in names])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True)
    row_labels = list(summary["label"])
    image = None
    matrices = [zero_shot, last_block]
    titles = ["(a) Zero-shot", "(b) After last_block fine-tuning"]
    for ax, matrix, title in zip(axes, matrices, titles, strict=True):
        image = draw_f1_heatmap(ax, np.array(matrix), row_labels, col_labels, hatched)
        ax.set_title(title, fontsize=MIN_FONT + 1, loc="left")
    fig.colorbar(image, ax=axes, label="F1 per class", fraction=0.025, pad=0.02)
    path = FIGURES_DIR / "ags_f1_per_class.pdf"
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return path


def draw_confusion(ax, counts: np.ndarray, names: list[str], title: str, show_ylabels: bool):
    """Counts in the cells, color = row-normalized value (recall)."""
    row_sums = counts.sum(axis=1, keepdims=True)  # (10, 1)
    recall = counts / np.maximum(row_sums, 1)
    ax.imshow(recall, cmap="Blues", vmin=0, vmax=1)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            color = "white" if recall[i, j] > 0.55 else INK
            ax.text(
                j, i, str(counts[i, j]), ha="center", va="center", fontsize=MIN_FONT, color=color
            )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=MIN_FONT)
    ax.set_yticks(range(len(names)))
    if show_ylabels:
        ax.set_yticklabels(names, fontsize=MIN_FONT)
        ax.set_ylabel("True class", fontsize=MIN_FONT)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted class", fontsize=MIN_FONT)
    ax.set_title(title, fontsize=MIN_FONT + 1)
    style_axes(ax)


def plot_confusion_selected(results: dict[str, dict], summary: pd.DataFrame, run_id: str) -> Path:
    """Confusion matrices of the selected run: zero-shot and last_block."""
    names = class_names_from(results)
    label = summary.set_index("run_id").loc[run_id, "label"]
    zero_shot = np.array(results[run_id]["ags"]["confusion"])
    last_block = np.array(results[run_id]["last_block"]["confusion"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 4.6))
    draw_confusion(ax_a, zero_shot, names, "(a) Zero-shot", True)
    draw_confusion(ax_b, last_block, names, "(b) After last_block", False)
    fig.suptitle(f"{label}: Aguascalientes test (color = row-normalized)", fontsize=MIN_FONT + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.94])  # leave room for the suptitle
    path = FIGURES_DIR / "confusion_selected.pdf"
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    return path


def plot_convergence(
    summary: pd.DataFrame, bands: str, y_limits: tuple[float, float], x_max: int
) -> Path:
    """Fig. 3: validation macro-F1 per epoch for the runs with the given bands."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for _, row in summary[summary["bands"] == bands].iterrows():
        log = pd.read_csv(LOGS_DIR / f"{row['run_id']}.csv")
        ax.plot(
            log["epoch"],
            log["val_macro_f1"],
            INIT_LINESTYLES[row["init"]],
            marker="o",
            markersize=3,
            color=MODEL_COLORS[row["model"]],
            label=f"{MODEL_NAMES[row['model']]} / {INIT_NAMES[row['init']]}",
        )
    ax.set_ylim(*y_limits)
    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_xlabel("Epoch", fontsize=MIN_FONT)
    ax.set_ylabel("Validation macro-F1", fontsize=MIN_FONT)
    ax.set_title(f"{BANDS_NAMES[bands]} input", fontsize=MIN_FONT + 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=MIN_FONT, loc="lower right", frameon=True)
    style_axes(ax)
    fig.tight_layout()
    path = FIGURES_DIR / f"convergence_{bands}.pdf"
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    return path


def convergence_limits(summary: pd.DataFrame) -> tuple[tuple[float, float], int]:
    """Common y-limits (with a small margin) and the longest run, over all training logs."""
    lowest = 1.0
    highest = 0.0
    longest = 1
    for run_id in summary["run_id"]:
        log = pd.read_csv(LOGS_DIR / f"{run_id}.csv")
        lowest = min(lowest, log["val_macro_f1"].min())
        highest = max(highest, log["val_macro_f1"].max())
        longest = max(longest, int(log["epoch"].max()))
    margin = 0.02
    return (max(0.0, lowest - margin), min(1.0, highest + margin)), longest


def write_figures(results: dict[str, dict], summary: pd.DataFrame, selected_run: str) -> list[Path]:
    """Write all the PDFs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.size"] = MIN_FONT
    plt.rcParams["pdf.fonttype"] = 42  # real fonts, not outlines
    written = [plot_drop_recovery(summary)]
    written.append(plot_f1_per_class(results, summary))
    written.append(plot_confusion_selected(results, summary, selected_run))
    y_limits, x_max = convergence_limits(summary)
    written.append(plot_convergence(summary, "rgb", y_limits, x_max))
    written.append(plot_convergence(summary, "ms13", y_limits, x_max))
    return written


# ---------------------------------------------------------------- main


def main() -> None:
    results = load_results(RESULTS_DIR)
    classes = macro_classes(results)
    summary = build_summary(results, LOGS_DIR)
    n_classes_total = len(class_names_from(results))

    written = write_tables(summary, classes, n_classes_total)

    print("Table: EuroSAT test (tab:eurosat)")
    print(eurosat_table(summary).to_string(index=False))
    print()
    print("Table: Aguascalientes test (tab:ags)")
    print(ags_table(summary).to_string(index=False))
    print()
    print("Table: cost (tab:cost)")
    print(cost_table(summary).to_string(index=False))
    print()

    all_runs = list(summary["run_id"])
    imagenet_runs = list(summary[summary["init"] == "imagenet"]["run_id"])
    selection_all = select_model(summary, all_runs)
    selection_imagenet = select_model(summary, imagenet_runs)
    print(format_selection("all 8 runs", selection_all, summary))
    print()
    print(format_selection("the 6 ImageNet runs", selection_imagenet, summary))
    print()

    # The text macros are read back from summary.csv, the same file the tables come from.
    summary_csv = pd.read_csv(TABLES_DIR / "summary.csv")
    macros = paper_macros(summary_csv, selection_all, selection_imagenet)
    macros.update(band_macros(band_zscores(AGS_CHECK_REPORT)))
    macros.update(share_macros(last_block_shares()))
    macros["FineTuneSteps"] = str(finetune_steps(BASE_CONFIG, AGS_LABELS))
    macros["HeadLossFallingRuns"] = str(head_loss_falling_runs(summary_csv, LOGS_DIR))
    written.append(write_macros(macros))

    written += write_figures(results, summary, selection_all["winner"])
    print("Written files:")
    for path in written:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
