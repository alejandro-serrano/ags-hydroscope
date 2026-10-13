"""Train one model on EuroSAT with the shared protocol, then evaluate it on the test split.

    python -m hydroscope.training.train --config configs/<run>.yaml

Same loop for every model (docs/protocol.md section 5): AdamW, 1 warmup epoch + cosine, batch 64
as 2 micro-batches of 32, at most 15 epochs, early stopping on val macro-F1.
"""

import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score

from hydroscope.config import load_config
from hydroscope.data.eurosat import CLASSES
from hydroscope.data.transforms import get_mean_std, resize_to_input
from hydroscope.models.factory import build_model, count_params
from hydroscope.training.evaluate import (
    evaluate_checkpoint,
    get_git_commit,
    get_git_dirty,
    predict,
    warn_if_dirty,
)
from hydroscope.training.loop_utils import (
    accumulation_steps,
    build_arg_parser,
    make_eurosat_loader,
)
from hydroscope.utils.device import PeakMemoryTracker, select_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (same as the course notebook)
    torch.manual_seed(seed)


def lr_at_step(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Learning rate of optimizer step ``step`` (0-based): linear warmup, then cosine to 0.

    The warmup reaches exactly ``base_lr`` at its last step (``warmup_steps - 1``); the cosine
    starts at ``base_lr`` and reaches 0 at ``total_steps``.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def run_epoch(
    model,
    loader,
    criterion,
    device,
    input_size,
    optimizer=None,
    accum_steps=1,
    state=None,
    trainable_modules=None,
):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates.

    ``state`` is a dict with the schedule and the memory tracker; it counts optimizer steps
    across epochs (keys: step, total_steps, warmup_steps, base_lr, max_steps, stop, tracker).
    ``trainable_modules`` (fine-tuning only) is the list of modules that learn; the rest of the
    model stays in eval mode. Returns the mean loss per sample, the predictions and the labels.
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    if training and trainable_modules is not None:
        # Frozen BatchNorm layers keep their EuroSAT statistics (and dropout stays off) when the
        # whole model is in eval mode; only the trainable modules go back to train mode.
        model.eval()
        for module in trainable_modules:
            module.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    if training:
        optimizer.zero_grad()
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)  # (N, C, 64, 64), (N,)
            x = resize_to_input(x, input_size)  # (N, C, 224, 224)
            logits = model(x)  # (N, 10)
            loss = criterion(logits, y)
            if training:
                # Gradient accumulation: the mean loss of the big batch is the mean of the
                # micro-batch losses, so divide by accum_steps and add the gradients up.
                (loss / accum_steps).backward()
                last_batch = i + 1 == len(loader)
                if (i + 1) % accum_steps == 0 or last_batch:
                    lr = lr_at_step(
                        state["step"], state["total_steps"], state["warmup_steps"], state["base_lr"]
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                    state["step"] += 1
                    state["tracker"].update()
            total_loss += loss.item() * len(y)
            all_preds += logits.argmax(dim=1).tolist()
            all_labels += y.tolist()
            if training and state["max_steps"] is not None and state["step"] >= state["max_steps"]:
                state["stop"] = True
                break
    return total_loss / len(all_labels), all_preds, all_labels


def epochs_to_95(val_f1_history: list[float]) -> int:
    """First epoch (1-based) whose val macro-F1 reaches 95 % of the best val macro-F1."""
    target = 0.95 * max(val_f1_history)
    for epoch, value in enumerate(val_f1_history, start=1):
        if value >= target:
            return epoch
    return len(val_f1_history)


def fit(model, train_loader, val_loader, cfg: dict, device, ckpt_path: Path, tracker) -> list[dict]:
    """Train with early stopping on val macro-F1. Saves the best weights and returns the history."""
    accum = accumulation_steps(cfg["optim"]["batch_size"], cfg["optim"]["micro_batch_size"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    state = {
        "step": 0,
        "total_steps": cfg["schedule"]["max_epochs"] * steps_per_epoch,
        "warmup_steps": cfg["schedule"]["warmup_epochs"] * steps_per_epoch,
        "base_lr": cfg["optim"]["lr"],
        "max_steps": cfg["schedule"]["max_steps"],
        "stop": False,
        "tracker": tracker,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    size = cfg["data"]["input_size"]
    history, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, cfg["schedule"]["max_epochs"] + 1):
        start = time.time()
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, size, optimizer, accum, state
        )
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, device, size)
        val_f1 = f1_score(
            val_labels, val_preds, average="macro", labels=list(range(10)), zero_division=0
        )
        seconds = time.time() - start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": accuracy_score(val_labels, val_preds),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": seconds,
            }
        )
        print(
            f"epoch {epoch:2d} | train loss {train_loss:.3f} | "
            f"val macro-F1 {val_f1:.3f} | {seconds:.0f}s"
        )
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
        if bad_epochs >= cfg["early_stopping"]["patience"] or state["stop"]:
            break
    return history


def write_checkpoint_meta(ckpt_path: Path, cfg: dict, mean, std, stats: dict) -> None:
    """Write ``<run>.yaml`` next to the checkpoint (contract 3 plus the training statistics)."""
    meta = {
        "run_id": cfg["run_id"],
        "model": cfg["model"],
        "bands": cfg["bands"],
        "init": cfg["init"],
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "classes": list(CLASSES),
        "protocol_version": cfg["protocol_version"],
    }
    meta.update(stats)
    meta["config"] = cfg  # evaluate needs data paths, results_dir and device
    ckpt_path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["schedule"]["max_steps"] = args.max_steps
    run_training(cfg)


def run_training(cfg: dict) -> None:
    """Train, save checkpoint + history, and evaluate on eurosat_test."""
    # Read the code version before training: this is the code that runs.
    git_commit = get_git_commit()
    git_dirty = get_git_dirty()
    warn_if_dirty(git_dirty)
    set_seed(cfg["seed"])
    device = select_device(cfg["device"])
    mean, std = get_mean_std(cfg["init"], cfg["bands"], cfg["data"]["stats"])
    train_loader = make_eurosat_loader(cfg, "train", mean, std, train=True)
    val_loader = make_eurosat_loader(cfg, "val", mean, std, train=False)

    tracker = PeakMemoryTracker(device)
    model = build_model(cfg["model"], cfg["bands"], cfg["init"]).to(device)
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['run_id']}.pt"

    print(
        f"run {cfg['run_id']} | {cfg['model']} {cfg['bands']} {cfg['init']} | device {device.type}"
    )
    start = time.time()
    history = fit(model, train_loader, val_loader, cfg, device, ckpt_path, tracker)
    train_time = time.time() - start

    logs_dir = Path(cfg["output"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(logs_dir / f"{cfg['run_id']}.csv", index=False)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))  # best epoch
    val_preds, val_labels, val_seconds = predict(
        model, val_loader, device, cfg["data"]["input_size"]
    )
    val_f1_history = [row["val_macro_f1"] for row in history]
    accum = accumulation_steps(cfg["optim"]["batch_size"], cfg["optim"]["micro_batch_size"])
    stats = {
        "train_time_s": train_time,
        "time_per_epoch_s": train_time / len(history),
        "peak_mem_mb": tracker.peak_mb,
        "epochs_to_95": epochs_to_95(val_f1_history),
        "micro_batch": cfg["optim"]["micro_batch_size"],
        "accum_steps": accum,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    write_checkpoint_meta(ckpt_path, cfg, mean, std, stats)
    print(
        f"params {count_params(model):,} | total time {train_time:.0f}s | "
        f"per epoch {stats['time_per_epoch_s']:.0f}s | peak memory {stats['peak_mem_mb']} MB | "
        f"val {len(val_labels) / val_seconds:.0f} patches/s | epochs to 95% {stats['epochs_to_95']}"
    )
    evaluate_checkpoint(ckpt_path, "eurosat_test", device)


if __name__ == "__main__":
    main()
