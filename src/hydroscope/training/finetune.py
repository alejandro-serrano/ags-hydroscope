"""Fine-tune a EuroSAT checkpoint on the Aguascalientes ``finetune`` split, then test it.

    python -m hydroscope.training.finetune --ckpt checkpoints/<run>.pt --mode head|last_block

Same optimizer, learning rate, weight decay, batch and augmentation as the EuroSAT training
(docs/protocol.md). Fixed number of epochs (``finetune.epochs``), no validation set: the last
epoch is the result. The ``test`` split is only read by the final evaluation.
"""

import argparse
import copy
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from hydroscope.config import load_config
from hydroscope.data.eurosat import CLASSES
from hydroscope.models.factory import (
    count_params,
    count_trainable_params,
    head_modules,
    last_block_modules,
)
from hydroscope.training.evaluate import (
    evaluate_checkpoint,
    get_git_commit,
    get_git_dirty,
    load_checkpoint,
    read_meta,
    warn_if_dirty,
)
from hydroscope.training.loop_utils import accumulation_steps, make_ags_loader
from hydroscope.training.train import run_epoch, set_seed
from hydroscope.utils.device import PeakMemoryTracker, select_device

MODES = ("head", "last_block")
SMOKE_CONFIG = Path("configs/smoke.yaml")


def class_weights(labels: list[int], num_classes: int = 10) -> torch.Tensor:
    """Inverse-frequency weights ``n / (num_classes * count_c)``, shape (num_classes,).

    Raises:
        ValueError: if a class has no patch (its weight would be infinite).
    """
    weights = []
    for class_id in range(num_classes):
        count = labels.count(class_id)
        if count == 0:
            raise ValueError(f"class {CLASSES[class_id]!r} has no patch in the fine-tuning split")
        weights.append(len(labels) / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def get_trainable_modules(model: nn.Module, model_name: str, mode: str) -> list[nn.Module]:
    """Modules that learn in each fine-tuning mode."""
    if mode == "head":
        return head_modules(model, model_name)
    if mode == "last_block":
        return last_block_modules(model, model_name)
    raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")


def freeze_all_but(model: nn.Module, trainable_modules: list[nn.Module]) -> None:
    """Set ``requires_grad=False`` everywhere, then ``True`` in ``trainable_modules``."""
    for param in model.parameters():
        param.requires_grad = False
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True


def apply_smoke_overrides(cfg: dict) -> dict:
    """Copy of ``cfg`` with the epochs and output folders of ``configs/smoke.yaml``."""
    smoke = load_config(SMOKE_CONFIG)
    cfg = copy.deepcopy(cfg)
    cfg["finetune"]["epochs"] = smoke["finetune"]["epochs"]
    cfg["output"] = smoke["output"]
    return cfg


def write_finetune_meta(ft_ckpt: Path, meta: dict, cfg: dict, mode: str, stats: dict) -> None:
    """Write ``<run>_ft-<mode>.yaml``: the parent metadata plus the fine-tuning statistics."""
    ft_meta = {
        "run_id": f"{meta['run_id']}_ft-{mode}",
        "model": meta["model"],
        "bands": meta["bands"],
        "init": meta["init"],
        "normalization": meta["normalization"],
        "classes": meta["classes"],
        "protocol_version": meta["protocol_version"],
        "parent_run_id": meta["run_id"],
        "finetune_mode": mode,
    }
    ft_meta.update(stats)
    ft_meta["config"] = cfg
    ft_ckpt.with_suffix(".yaml").write_text(yaml.safe_dump(ft_meta, sort_keys=False))


def run_finetune(ckpt_path: Path, mode: str, smoke: bool = False) -> dict:
    """Fine-tune ``ckpt_path`` in ``mode``, save the last weights and evaluate on ``ags_test``.

    Returns the ``ags_test`` result record (also written as JSON by ``evaluate_checkpoint``).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    git_commit = get_git_commit()
    git_dirty = get_git_dirty()
    warn_if_dirty(git_dirty)

    meta = read_meta(ckpt_path)
    cfg = meta["config"]
    if smoke:
        cfg = apply_smoke_overrides(cfg)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    mean = torch.tensor(meta["normalization"]["mean"])  # (C,)
    std = torch.tensor(meta["normalization"]["std"])  # (C,)
    run_id = f"{meta['run_id']}_ft-{mode}"

    model = load_checkpoint(ckpt_path, meta, device)
    trainable_modules = get_trainable_modules(model, meta["model"], mode)
    freeze_all_but(model, trainable_modules)
    trainable_params = count_trainable_params(model)

    train_loader = make_ags_loader(cfg, "finetune", mean, std, train=True)
    labels = train_loader.dataset.labels
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels).to(device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    accum = accumulation_steps(cfg["optim"]["batch_size"], cfg["optim"]["micro_batch_size"])
    epochs = cfg["finetune"]["epochs"]
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    tracker = PeakMemoryTracker(device)
    state = {
        "step": 0,
        "total_steps": epochs * steps_per_epoch,
        "warmup_steps": cfg["schedule"]["warmup_epochs"] * steps_per_epoch,
        "base_lr": cfg["optim"]["lr"],
        "max_steps": None,
        "stop": False,
        "tracker": tracker,
    }
    size = cfg["data"]["input_size"]
    print(
        f"finetune {run_id} | mode {mode} | {len(labels)} patches | "
        f"trainable {trainable_params:,} of {count_params(model):,} | device {device.type}"
    )

    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, preds, epoch_labels = run_epoch(
            model, train_loader, criterion, device, size, optimizer, accum, state, trainable_modules
        )
        train_f1 = f1_score(
            epoch_labels, preds, average="macro", labels=list(range(10)), zero_division=0
        )
        seconds = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": seconds,
            }
        )
        print(
            f"epoch {epoch:2d} | train loss {train_loss:.3f} | "
            f"train macro-F1 {train_f1:.3f} | {seconds:.0f}s"
        )
    train_time = time.time() - start

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    logs_dir = Path(cfg["output"]["logs_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(logs_dir / f"{run_id}.csv", index=False)
    ft_ckpt = ckpt_dir / f"{run_id}.pt"
    torch.save(model.state_dict(), ft_ckpt)  # last epoch, no selection

    stats = {
        "train_time_s": train_time,
        "time_per_epoch_s": train_time / epochs,
        "peak_mem_mb": tracker.peak_mb,
        "epochs_to_95": None,  # no validation set in fine-tuning
        "micro_batch": cfg["optim"]["micro_batch_size"],
        "accum_steps": accum,
        "trainable_params": trainable_params,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    write_finetune_meta(ft_ckpt, meta, cfg, mode, stats)
    print(
        f"params {count_params(model):,} | trainable {trainable_params:,} | "
        f"total time {train_time:.0f}s | per epoch {stats['time_per_epoch_s']:.0f}s | "
        f"peak memory {stats['peak_mem_mb']} MB | "
        f"train {epochs * len(labels) / train_time:.0f} patches/s"
    )
    return evaluate_checkpoint(ft_ckpt, "ags_test", device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a checkpoint on Aguascalientes")
    parser.add_argument("--ckpt", required=True, help="path to the EuroSAT checkpoint .pt")
    parser.add_argument("--mode", required=True, help="head | last_block")
    parser.add_argument(
        "--smoke", action="store_true", help="epochs and output folders of configs/smoke.yaml"
    )
    args = parser.parse_args()
    run_finetune(Path(args.ckpt), args.mode, args.smoke)


if __name__ == "__main__":
    main()
