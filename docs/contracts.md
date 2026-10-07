# Data contracts

Formats exchanged between pipeline stages (protocol v1.0). Producers must write exactly these
formats; consumers may rely on them. Changing a contract requires updating this file, every producer
and consumer, and `src/hydroscope/contracts.py` in the same PR.

Data, rasters and checkpoints are never committed; they are published as GitHub Release assets.

## 1. Patch

- **Path:** `data/ags/patches/{cell_id}.tif`
- **Format:** GeoTIFF, one patch per 640 m grid cell.

| Property | Value |
| --- | --- |
| Shape | 13 × 64 × 64 (bands × rows × cols) |
| Dtype | `uint16` |
| CRS | EPSG:32613 |
| Band order | TorchGeo EuroSAT: B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A (B8A last) |
| Values | TOA reflectance × 10000 |

- **Written by:** `hydroscope.geo` tiling (from the Earth Engine export).
- **Read by:** labeling notebook (`labeling/`), Aguascalientes dataset (`hydroscope.data`),
  evaluation, fine-tuning and Grad-CAM.

## 2. Labels

- **Path:** `data/ags/labels.csv`
- **Format:** CSV, UTF-8, header row, one row per labeled cell.

| Column | Type | Description |
| --- | --- | --- |
| `cell_id` | str | Grid cell id; matches the patch filename |
| `label` | str | One of the 10 class names (protocol section 3) |
| `annotator` | str | Annotator id |
| `block_id` | str | 5×5 km block containing the cell |
| `split` | str | `finetune` or `test` |

Invariants: `cell_id` is unique; every `block_id` appears in exactly one `split`.

- **Written by:** labeling notebook (`label`, `annotator`) and the spatial split script
  (`block_id`, `split`).
- **Read by:** Aguascalientes dataset, fine-tuning, evaluation, `experiment-auditor`.

## 3. Checkpoint

- **Path:** `checkpoints/{model}_{bands}_{init}.pt` plus `checkpoints/{model}_{bands}_{init}.yaml`
- **Format:** `.pt` = PyTorch `state_dict` (`torch.save`); `.yaml` = metadata below.

| `.yaml` field | Type | Description |
| --- | --- | --- |
| `model` | str | `resnet50` · `efficientnet_b0` · `vit_small` |
| `bands` | str | `rgb` · `ms13` |
| `init` | str | `imagenet` · `ssl4eo` |
| `normalization` | object | `mean`: list[float], `std`: list[float], one value per input band, in band order |
| `classes` | list[str] | Class names in model output order |
| `protocol_version` | str | Protocol version used for training (e.g. `"1.0"`) |

- **Written by:** `hydroscope.training.train` (and `finetune`).
- **Read by:** `hydroscope.training.evaluate`, `hydroscope.training.finetune`, Grad-CAM.

## 4. Result

- **Path:** `experiments/results/{run_id}.json`
- **Format:** JSON object, flat. All fields are always present; fields marked nullable are `null`
  when they do not apply.

| Field | Type | Description |
| --- | --- | --- |
| `run_id` | str | Unique id of the run; matches the file name |
| `model` | str | `resnet50` · `efficientnet_b0` · `vit_small` |
| `bands` | str | `rgb` · `ms13` |
| `init` | str | `imagenet` · `ssl4eo` |
| `split` | str | Evaluated split, e.g. `eurosat_test`, `ags_test` |
| `accuracy` | float | Overall accuracy, 0–1 |
| `macro_f1` | float | Macro-F1, 0–1 (on Aguascalientes, excludes classes with < 10 test patches) |
| `f1_per_class` | dict[str, float] | F1 for every class, keyed by class name |
| `confusion` | list[list[int]] | Confusion matrix; rows = true, cols = predicted, class index order |
| `params` | int | Total number of parameters |
| `train_time_s` | float, nullable | Total training time in seconds; `null` for evaluation-only runs |
| `peak_mem_mb` | float, nullable | Peak GPU memory in MB; `null` on CPU |
| `epochs_to_95` | int, nullable | First epoch reaching 95 % of the best val macro-F1; `null` for evaluation-only runs |
| `patches_per_s` | float | Inference throughput on the evaluated split |
| `protocol_version` | str | Protocol version (e.g. `"1.0"`) |
| `git_commit` | str | Commit hash of the code that produced the file |

Validated by `hydroscope.contracts.validate_result`.

- **Written by:** `hydroscope.training.train`, `evaluate` and `finetune` only; never edited by hand.
- **Read by:** `scripts/make_tables.py`, `scripts/make_figures.py`, `experiment-auditor`.
