# Experimental protocol

**Status:** frozen · **Protocol version:** 1.0

This protocol is shared by every model. ResNet-50, EfficientNet-B0 and ViT-S/16 differ **only** in
the architecture. The machine-readable copy is `configs/base.yaml`; `tests/test_config.py` checks
that both agree and that run configs do not override shared keys.

**Rule:** never change a hyperparameter for only one model. If something must change, change it in
`configs/base.yaml` for all models and add a row to the [change log](#9-change-log) in the same
commit.

## 1. Research questions

- **RQ-a.** Which of ResNet-50, EfficientNet-B0 and ViT-S/16 performs best on EuroSAT, and at what
  computational cost?
- **RQ-b.** How much does each model degrade on Aguascalientes zero-shot?
- **RQ-c.** How much of that degradation does light fine-tuning recover?
- **RQ-d.** Does 13-band input outperform RGB?
- **RQ-e.** Do SSL4EO-S12 weights outperform ImageNet weights?

## 2. Data

- **EuroSAT:** EuroSAT MS from TorchGeo with its official train/val/test split. Band order is
  TorchGeo's: B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B8A (B8A last).
  RGB input = B04, B03, B02 taken from the same files.
- **Aguascalientes:** 1,102 labeled patches from April 2024 (1,270 cells reviewed, 168
  skipped). Spatial split by blocks of 8×8 grid cells (5.12 km): 50 % fine-tune, 50 % test,
  stratified by class. No block appears in both splits.
  Method (`scripts/split_ags.py`): the unit is the block (a whole block goes to one split);
  with `numpy.random.default_rng(0)` the block list is shuffled 20,000 times and cut half/half
  (first half fine-tune, second half test); the partition with the smallest maximum deviation,
  across classes, of the fraction of the class's patches in test from 0.5 is kept (min-max
  criterion; `skip` rows are not counted and take the split of their block). Seed: 0. Outputs:
  `split` column of `data/ags/labels.csv` and `configs/ags_block_split.csv` (block_id,split).
  Counts per class and split: `docs/data_card.md`.

## 3. Classes

The 10 EuroSAT classes, in TorchGeo index order:

| Index | Class |
| --- | --- |
| 0 | AnnualCrop |
| 1 | Forest |
| 2 | HerbaceousVegetation |
| 3 | Highway |
| 4 | Industrial |
| 5 | Pasture |
| 6 | PermanentCrop |
| 7 | Residential |
| 8 | River |
| 9 | SeaLake |

Classes with fewer than 10 test patches in Aguascalientes are excluded from its macro-F1 and are
reported separately.

## 4. Models and initialization

| Config name | timm model | Weights |
| --- | --- | --- |
| `resnet50` | `resnet50` | ImageNet |
| `efficientnet_b0` | `efficientnet_b0` | ImageNet |
| `vit_small` | `vit_small_patch16_224` | ImageNet |

- **13-band input:** copy the RGB first-conv weights to the B04, B03, B02 input channels and
  initialize every other channel with their mean.
- **SSL variant (priority: Should), 13 bands only:** TorchGeo
  `ResNet50_Weights.SENTINEL2_ALL_MOCO` and `ViTSmall16_Weights.SENTINEL2_ALL_DINO`.
  The SSL4EO-S12 weights use the band order B01…B08, B8A, B09…B12 (B8A after B08), so the factory
  permutes the input channels of their first layer to EuroSAT order (B8A last); the data pipeline
  keeps EuroSAT order for every model. Normalization for these weights is TorchGeo's `x / 10000`.
- **Last block** (fine-tuning mode `last_block`, section 6): ResNet-50 `layer4` + `fc`;
  EfficientNet-B0 `blocks[-1]` + `conv_head` + `bn2` + `classifier`; ViT-S/16 `blocks[-1]` +
  `norm` + `head`. Implemented by `last_block_params` in `src/hydroscope/models/factory.py`.
  Note: the share of parameters trained in `last_block` mode differs by architecture (ResNet-50
  ≈64 %, EfficientNet-B0 ≈28 %, ViT-S/16 ≈8 % of total, from `count_params`); this asymmetry is
  inherent to the architectures, is kept as is, and is discussed in the paper.

## 5. Shared training configuration

| Setting | Value |
| --- | --- |
| Input | 64×64 resized to 224×224 (bilinear) |
| Normalization | Per-band, from EuroSAT train (TorchGeo's normalization for SSL4EO weights) |
| Augmentation | Horizontal and vertical flips + 90° rotations |
| Optimizer | AdamW, lr 1e-4, weight decay 0.05 |
| Schedule | 1 warmup epoch + cosine |
| Batch size | 64 (gradient accumulation if needed) |
| Epochs | Max 15 |
| Early stopping | Patience 3 on val macro-F1 |
| Loss | Cross-entropy (class-weighted on Aguascalientes) |
| Seed | 0 |
| Hardware | Same GPU for all runs |

## 6. Experiment matrix

| ID | Description | Runs |
| --- | --- | --- |
| E1 | 3 models × {RGB, 13 bands}, ImageNet weights | 6 |
| E2 | SSL4EO-S12 weights (ResNet-50 MoCo, ViT-S/16 DINO), 13 bands | 2 |
| E3 | Zero-shot evaluation of the 8 checkpoints on Aguascalientes | 8 evaluations |
| E4 | Fine-tuning of the 8 checkpoints, modes `head` and `last_block`, 10 fixed epochs | 16 |

- `head`: backbone frozen, classification head trained. Head modules: ResNet-50 `fc`,
  EfficientNet-B0 `classifier`, ViT-S/16 `head`.
- `last_block`: last stage/block and head trained (modules listed in section 4).
- Frozen parts run in eval mode while fine-tuning: BatchNorm keeps its EuroSAT statistics and
  dropout is off. Only the trainable modules are in train mode.
- Fixed 10 epochs; the weights of the last epoch are the result. There is no validation set, no
  early stopping and no model selection; the `test` split is never seen during fine-tuning.
- Loss: cross-entropy with class weights `n / (K * n_c)` (`n` patches, `K = 10` classes, `n_c`
  patches of class `c`), computed on the fine-tune split only.
- Same AdamW (lr 1e-4, weight decay 0.05), batch 64 (2 x 32), augmentation and seed as in
  section 5; schedule: 1 warmup epoch + cosine over the 10 fine-tuning epochs.
- Aguascalientes macro-F1 averages the classes with at least 10 test patches (today River, with 7,
  is excluded). The result also reports `macro_f1_classes` (the classes used) and
  `macro_f1_all_classes` (macro-F1 over the 10 classes).
- `epochs_to_95` is `null` for fine-tuned results (no validation curve).

## 7. Metrics

- **Primary:** macro-F1.
- Also: accuracy, per-class F1, confusion matrix, number of parameters, total and per-epoch
  training time, peak GPU memory, throughput (patches/s), epochs to 95 % of the best val macro-F1.
- **Drop** = F1<sub>EuroSAT</sub> − F1<sub>Ags, zero-shot</sub>
- **Recovery** = (F1<sub>Ags, fine-tuned</sub> − F1<sub>Ags, zero-shot</sub>) / Drop

F1<sub>EuroSAT</sub> is the macro-F1 over the same classes as the Aguascalientes macro-F1, computed from
`f1_per_class` of the `eurosat_test` result. Recovery is not defined when Drop ≤ 0.

Every number in the paper, slides and README is generated by a script from
`experiments/results/*.json`.

## 8. Selection rule

The selected model is the one with the highest Aguascalientes macro-F1 after `last_block`
fine-tuning. If two models are within 1 point (≤ 0.01 macro-F1), the one with higher patches/s wins.
The tie-break uses the `eurosat_test` patches/s, the same number as the cost table.

## 9. Change log

| Date | Change | Reason | Applies to all models |
| --- | --- | --- | --- |
| 2026-10-07 | Protocol frozen (v1.0) | — | yes |
| 2026-10-07 | Hardware: Apple M2 Pro (MPS), fp32; effective batch 64 = 2 × 32 with gradient accumulation (BatchNorm sees 32 samples); peak memory = MPS driver memory; data cached as .npy; resize on device | Only machine available | yes |
| 2026-10-08 | Spatial blocks of 8×8 cells (5.12 km) instead of 5×5 km, so that every cell belongs to exactly one block | Alignment with the 640 m grid | yes |
| 2026-10-12 | Aguascalientes split method fixed: 20,000 random half/half block partitions, min-max class deviation, seed 0 | Reproducible, class-balanced spatial split | yes |
| 2026-10-13 | Fine-tuning details (E4) written down: head modules per architecture; frozen parts in eval mode; fixed 10 epochs, last epoch used, no validation, test never seen; class weights `n/(K·n_c)` from the fine-tune split; same AdamW/lr/wd/batch/augmentation/seed with 1 warmup epoch + cosine over 10 epochs | Clarification before running E4 (protocol version stays 1.0) | yes |
| 2026-10-13 | Aguascalientes macro-F1 class set fixed: classes with at least 10 test patches (River excluded today); `macro_f1_all_classes` also reported; `epochs_to_95 = null` for fine-tuning | Clarification of section 3 before running E3/E4 (protocol version stays 1.0) | yes |
| 2026-10-13 | Clarification: F1 EuroSAT in Drop/Recovery is the macro-F1 over the Aguascalientes class set; the tie-break uses `eurosat_test` patches/s | Make the computation in `scripts/make_results.py` unambiguous (protocol version stays 1.0) | yes |
