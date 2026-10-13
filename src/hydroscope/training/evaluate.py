"""Evaluate a checkpoint on a split and write the result JSON (contract 4 in docs/contracts.md).

python -m hydroscope.training.evaluate --ckpt checkpoints/<run>.pt --split eurosat_test
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from hydroscope.contracts import validate_result
from hydroscope.data.eurosat import CLASSES
from hydroscope.data.transforms import resize_to_input
from hydroscope.models.factory import build_model, count_params
from hydroscope.training.loop_utils import make_eurosat_loader
from hydroscope.utils.device import select_device

EUROSAT_SPLITS = {"eurosat_val": "val", "eurosat_test": "test"}
"""Result split name -> EuroSAT split name."""

REPO_ROOT = Path(__file__).resolve().parents[3]


def get_git_commit() -> str:
    """Hash of the current commit, or "unknown" if git is not available."""
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
        return output.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_git_dirty() -> bool:
    """True if src/ or configs/ have uncommitted changes (new files included).

    Only the code and the configs decide a result, so changes elsewhere (docs, notes) do not
    count. If git is not available we cannot check, so we return True to be safe.
    """
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "configs"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        return output.stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return True


def warn_if_dirty(git_dirty: bool) -> None:
    """Print a visible warning when the code has uncommitted changes (it does not stop the run)."""
    if git_dirty:
        print(
            "WARNING: uncommitted changes in src/ or configs/ (git_dirty = true); git_commit "
            "does not describe this code exactly. Commit before the final runs."
        )


def read_meta(ckpt_path: Path) -> dict:
    """Read the checkpoint .yaml (contract 3) that sits next to ``<run>.pt``."""
    return yaml.safe_load(Path(ckpt_path).with_suffix(".yaml").read_text())


def load_checkpoint(ckpt_path: Path, meta: dict, device: torch.device) -> nn.Module:
    """Build the architecture described by ``meta``, load the ``.pt`` weights, set eval mode."""
    # The architecture is all we need from build_model: the weights are replaced right below.
    model = build_model(meta["model"], meta["bands"], meta["init"])
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def predict(model, loader, device: torch.device, input_size: int):
    """Predict a whole loader. Returns ``(preds, labels, seconds)`` (seconds = inference time)."""
    model.eval()
    all_preds, all_labels = [], []
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)  # (N, C, 64, 64)
            x = resize_to_input(x, input_size)  # (N, C, 224, 224)
            logits = model(x)  # (N, 10)
            all_preds += logits.argmax(dim=1).cpu().tolist()
            all_labels += y.tolist()
    return all_preds, all_labels, time.time() - start


def compute_metrics(labels: list[int], preds: list[int]) -> dict:
    """Accuracy, macro-F1, F1 per class (keyed by class name) and the 10x10 confusion matrix."""
    class_ids = list(range(len(CLASSES)))
    f1_values = f1_score(labels, preds, average=None, labels=class_ids, zero_division=0)
    f1_per_class = {}
    for name, value in zip(CLASSES, f1_values, strict=True):
        f1_per_class[name] = float(value)
    confusion = confusion_matrix(labels, preds, labels=class_ids)  # rows = true, cols = predicted
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(
            f1_score(labels, preds, average="macro", labels=class_ids, zero_division=0)
        ),
        "f1_per_class": f1_per_class,
        "confusion": confusion.tolist(),
    }


def evaluate_checkpoint(ckpt_path: Path, split: str, device: torch.device | None = None) -> dict:
    """Evaluate a checkpoint on ``split``, validate the record and write the result JSON.

    The JSON goes to ``{results_dir}/{run_id}_{split}.json`` (results_dir from the run config
    stored in the checkpoint .yaml). Training fields come from that .yaml.
    """
    if split == "ags_test":
        raise NotImplementedError("the Aguascalientes split (ags_test) arrives in F4")
    if split not in EUROSAT_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(EUROSAT_SPLITS)}")
    meta = read_meta(ckpt_path)
    cfg = meta["config"]
    if device is None:
        device = select_device(cfg["device"])
    model = load_checkpoint(ckpt_path, meta, device)
    mean = torch.tensor(meta["normalization"]["mean"])
    std = torch.tensor(meta["normalization"]["std"])
    loader = make_eurosat_loader(cfg, EUROSAT_SPLITS[split], mean, std, train=False)
    preds, labels, seconds = predict(model, loader, device, cfg["data"]["input_size"])

    record = {
        "run_id": f"{meta['run_id']}_{split}",
        "model": meta["model"],
        "bands": meta["bands"],
        "init": meta["init"],
        "split": split,
    }
    record.update(compute_metrics(labels, preds))
    record["params"] = count_params(model)
    record["train_time_s"] = meta["train_time_s"]  # from the checkpoint .yaml
    record["peak_mem_mb"] = meta["peak_mem_mb"]
    record["epochs_to_95"] = meta["epochs_to_95"]
    record["patches_per_s"] = len(labels) / seconds
    record["device"] = device.type
    record["micro_batch"] = meta["micro_batch"]
    record["accum_steps"] = meta["accum_steps"]
    record["protocol_version"] = meta["protocol_version"]
    record["git_commit"] = get_git_commit()
    record["git_dirty"] = get_git_dirty()
    warn_if_dirty(record["git_dirty"])
    validate_result(record)

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{record['run_id']}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{split}: macro-F1 {record['macro_f1']:.3f} | accuracy {record['accuracy']:.3f}")
    print(f"wrote {out_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and write the result JSON")
    parser.add_argument("--ckpt", required=True, help="path to the checkpoint .pt")
    parser.add_argument("--split", default="eurosat_test", help="eurosat_val | eurosat_test")
    args = parser.parse_args()
    evaluate_checkpoint(Path(args.ckpt), args.split)


if __name__ == "__main__":
    main()
