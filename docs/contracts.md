# Data contracts

Formats exchanged between pipeline stages. Producers must write exactly these formats; consumers
may rely on them. Changing a contract requires updating this file and every producer/consumer in
the same PR.

Data, rasters and checkpoints are never committed; they are published as GitHub Release assets.

## 1. Patch GeoTIFF

Path: `data/ags/patches/<cell_id>.tif` (+ RGB preview `data/ags/previews/<cell_id>.png`).

| Property | Value |
| --- | --- |
| Shape | 13 × 64 × 64 (bands × rows × cols) |
| Dtype | `uint16`, DN = reflectance × 10000 (Sentinel-2 L1C harmonized) |
| CRS / resolution | EPSG:32613, 10 m (one 640 m grid cell) |
| Band order | B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A |
| Preview | RGB = B04, B03, B02, 2–98 % stretch, 8-bit PNG |

## 2. `labels.csv`

Path: `data/ags/labels.csv`. One row per labeled cell, UTF-8, header row.

| Column | Type | Description |
| --- | --- | --- |
| `cell_id` | str | Stable grid cell id; matches the patch filename |
| `label` | str | One of the 10 EuroSAT class names (below) |
| `annotator` | str | Annotator initials |
| `block_id` | str | 5×5 km block containing the cell |
| `split` | str | `train` · `val` · `test` |

Invariants: `cell_id` is unique; every `block_id` appears in exactly one `split`.

Classes (EuroSAT order): AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake.

## 3. Checkpoint

Path: `checkpoints/<run_id>.pt` + `checkpoints/<run_id>.yaml`.

- `.pt`: `torch.save` of a dict with `model_state` (state dict), `run_id`, `model`, `bands`,
  `init`, `epoch` (best), `val_macro_f1` (best), `classes` (list, order above).
- `.yaml`: the fully resolved config used for the run (after `extends`).

## 4. Result JSON

Path: `experiments/results/<run_id>__<stage>__<split>.json`. Written only by
`hydroscope.training.{train,evaluate,finetune}`; never edited by hand.

| Field | Type | Description |
| --- | --- | --- |
| `run_id` | str | Matches the config and checkpoint |
| `model` | str | `resnet50` · `efficientnet_b0` · `vit_small` |
| `bands` | str | `rgb` · `ms13` |
| `init` | str | `imagenet` · `ssl4eo` |
| `stage` | str | `pretrain` (EuroSAT) · `zero_shot` · `head` · `last_block` |
| `split` | str | `eurosat_test` · `ags_test` · … |
| `seed` | int | Random seed |
| `git_commit` | str | Commit hash of the code that produced the file |
| `created_at` | str | ISO 8601 timestamp (UTC) |
| `config` | object | Fully resolved config |
| `classes` | list[str] | Class order for per-class metrics and the confusion matrix |
| `metrics.accuracy` | float | Overall accuracy, 0–1 |
| `metrics.macro_f1` | float | Macro-averaged F1, 0–1 (main metric) |
| `metrics.per_class_f1` | object | `{class: f1}` |
| `metrics.confusion_matrix` | list[list[int]] | Rows = true, cols = predicted, in `classes` order |
| `metrics.n_samples` | int | Number of evaluated patches |
| `cost.params` | int | Total parameters |
| `cost.trainable_params` | int | Trainable parameters in this stage |
| `cost.train_time_s` | float | Total training wall time (null for evaluation-only stages) |
| `cost.time_per_epoch_s` | float | Mean epoch wall time |
| `cost.peak_gpu_mem_bytes` | int | `torch.cuda.max_memory_allocated` (null on CPU) |
| `cost.patches_per_s` | float | Inference throughput on the evaluated split |
| `convergence.epochs_trained` | int | Epochs actually run |
| `convergence.best_epoch` | int | Epoch of the best val macro-F1 |
| `convergence.epochs_to_95pct_best` | int | First epoch with val macro-F1 ≥ 0.95 × best |
| `convergence.history` | list[object] | Per epoch: `epoch`, `train_loss`, `val_loss`, `val_macro_f1`, `epoch_time_s` |

Fields that do not apply to a stage (e.g. training cost for `zero_shot`) are `null`, never omitted.
