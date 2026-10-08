# ags-hydroscope

Comparative study of CNNs vs. a Vision Transformer for Sentinel-2 land-use classification,
pre-trained on EuroSAT and evaluated on Aguascalientes, Mexico (semi-arid domain shift).
Final project for the Deep Learning course. Deadline: **Wed 2026-10-21**.

Stage 1 (graded): models, generalization, Grad-CAM, IEEE paper, slides, reproducible code.
Stage 2 (after delivery): monthly 2017–2026 processing, 3 municipal indicators, bilingual web app.

Brand: the app is always called **Hydroscope Aguascalientes** (never translated). Only the subtitle
changes with the language: "Monitor del agua de Aguascalientes" (es) / "Aguascalientes Water Monitor" (en).

## Language
- Code, comments, docstrings, commit messages, README, paper and slides: **English**.
- Web app UI (stage 2): Spanish and English via i18n keys. Never hard-code UI strings.
- You may answer the user in Spanish when they write in Spanish.

## Environment and commands
- Conda env: `conda env create -f environment.yml && conda activate hydroscope`
- Lint/format: `ruff check . --fix && ruff format .`
- Tests: `python -m pytest -q`
- Train: `python -m hydroscope.training.train --config configs/<run>.yaml`
- Evaluate: `python -m hydroscope.training.evaluate --ckpt checkpoints/<run>.pt --split ags_test`
- Fine-tune: `python -m hydroscope.training.finetune --ckpt checkpoints/<run>.pt --mode head|last_block`
- Smoke test (CPU, 200 images): `python -m hydroscope.training.train --config configs/smoke.yaml`

## Layout
- `src/hydroscope/data/` EuroSAT + Aguascalientes datasets, transforms
- `src/hydroscope/models/factory.py` ResNet-50, EfficientNet-B0, ViT-S/16 (3 or 13 bands)
- `src/hydroscope/training/` train, evaluate, finetune
- `src/hydroscope/explain/gradcam.py`
- `src/hydroscope/geo/` Earth Engine export, 640 m grid, tiling (stage 2: inference, indicators)
- `configs/` one YAML per run · `experiments/results/` one JSON per run (script output only)
- `paper/` IEEE LaTeX · `slides/` · `docs/` protocol, contracts, data card, model card

## Experimental protocol — invariants (see @docs/protocol.md)
- The three models share EVERYTHING except the architecture: input 224×224 (resized from 64),
  per-band normalization from EuroSAT train, flips + 90° rotations, AdamW lr 1e-4, wd 0.05,
  1 warmup epoch + cosine, batch 64, max 15 epochs, early stopping patience 3 on val macro-F1, seed 0.
- Never change a hyperparameter for only one model. If something must change, change it for all
  and record it in `docs/protocol.md` in the same commit.
- 13-band input: copy ImageNet RGB first-conv weights to B04/B03/B02, mean of them for the rest.
- Band order = TorchGeo EuroSAT order (B01…B12, B8A last). RGB = B04, B03, B02.
- Aguascalientes split is spatial (8×8-cell (5.12 km) blocks). Never move a cell between splits.
- Main metric: macro-F1. Also accuracy, per-class F1, confusion matrix, params, train time,
  peak GPU memory, patches/s, epochs to 95 % of best val macro-F1.

## Data contracts (see @docs/contracts.md)
Patch GeoTIFF 13×64×64 uint16 · `labels.csv` (cell_id, label, annotator, block_id, split) ·
checkpoint `.pt` + `.yaml` · result JSON with the fields listed in the contract.

## Hard rules
- **Never type, round or invent a metric.** Every number in the paper, slides or README must be
  generated from `experiments/results/*.json` by a script. Files in `experiments/results/` are
  written only by scripts (editing them is denied in settings).
- Never commit data, rasters or checkpoints; they go to GitHub Releases.
- Never read or print `.env` or credentials.
- Long training runs: write the command for the user to launch; do not start multi-hour jobs
  unless asked.
- Prefer small, reviewable PRs, one issue each, branch name `<task-id>-<slug>` (e.g. `F3-03-train-loop`).
- Stage 2 work lives on `stage2/*` branches and must never block stage 1 checkpoints.

## Style
- Python 3.11, type hints, docstrings on public functions, pure functions where possible.
- Tests for transforms, tiling, model factory shapes and (stage 2) indicators.
- Figures: matplotlib, saved as PDF in `paper/figures/` by scripts in `notebooks/` or `scripts/`.
