"""Run E3 and E4 on Aguascalientes: 8 zero-shot evaluations, then 16 fine-tunings.

    python scripts/run_ags.py

Run it from the repository root. A job whose result JSON already exists is skipped. A job
that fails prints its error and the script goes on with the next one. Commit before running,
so the results have git_dirty = false.
"""

import gc
import sys
from pathlib import Path

import torch

from hydroscope.config import load_config
from hydroscope.training.evaluate import evaluate_checkpoint
from hydroscope.training.finetune import MODES, run_finetune

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
NOT_RUNS = ("base", "smoke")


def list_run_ids() -> list[str]:
    """Run ids of the 8 EuroSAT checkpoints, from the YAML files in configs/."""
    run_ids = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        if path.stem not in NOT_RUNS:
            run_ids.append(path.stem)
    return run_ids


def free_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    run_ids = list_run_ids()
    jobs = []  # (job name, checkpoint, mode or None, result JSON path)
    for run_id in run_ids:
        jobs.append((run_id, None, f"{run_id}_ags_test"))
    for run_id in run_ids:
        for mode in MODES:
            jobs.append((run_id, mode, f"{run_id}_ft-{mode}_ags_test"))

    done, skipped, failed = [], [], []
    for run_id, mode, result_name in jobs:
        cfg = load_config(CONFIGS_DIR / f"{run_id}.yaml")
        result_path = Path(cfg["output"]["results_dir"]) / f"{result_name}.json"
        if result_path.exists():
            print(f"skip {result_name} (already exists)")
            skipped.append(result_name)
            continue
        ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{run_id}.pt"
        print(f"=== {result_name}")
        try:
            if mode is None:
                evaluate_checkpoint(ckpt_path, "ags_test")
            else:
                run_finetune(ckpt_path, mode)
            done.append(result_name)
        except Exception as error:
            print(f"FAILED {result_name}: {error}")
            failed.append(result_name)
        free_memory()

    print(f"summary: {len(done)} done | {len(skipped)} skipped | {len(failed)} failed")
    for name in failed:
        print(f"  failed: {name}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
