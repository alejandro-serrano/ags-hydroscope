# Hydroscope Aguascalientes

**Final project of the Deep Learning module** · Master's in Artificial Intelligence and Data
Science, Universidad Panamericana, Campus Aguascalientes.

Can a land-use classifier trained on European satellite images work in semi-arid
Aguascalientes, Mexico? This repository trains two CNNs (ResNet-50 and EfficientNet-B0) and a
Vision Transformer (ViT-S/16) on EuroSAT, tests them on Sentinel-2 patches of Aguascalientes
that we labeled by hand, and measures how much they lose there and how much a light
fine-tuning wins back.

Every table and every number of the study comes from the result files committed here, so
you can check them without training anything.

[![CI](https://github.com/alejandro-serrano/ags-hydroscope/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

---

## What the study compares

| Piece | Choice |
|---|---|
| Models | ResNet-50, EfficientNet-B0 and ViT-S/16, from timm |
| Variants | RGB or 13 bands · ImageNet or SSL4EO-S12 weights (self-supervised, learned on Sentinel-2) · eight runs |
| Training | EuroSAT with its official TorchGeo split, 10 classes, one shared protocol ([docs/protocol.md](docs/protocol.md)) |
| Target | Aguascalientes, Sentinel-2 L1C, April 2024 median; 640 m cells labeled by hand and split by blocks of 8×8 cells (5.12 km) |
| Fine-tuning | `head` (classifier only) or `last_block` (last block and classifier), with a class-weighted loss |
| Metrics | Macro-F1 (main), accuracy, drop and recovery, cost, convergence and Grad-CAM |

The data are described in [docs/data_card.md](docs/data_card.md) and the selected model in
[docs/model_card.md](docs/model_card.md).

---

## How it is organized

```
src/hydroscope/   data, models, training, explain, geo
configs/          one YAML per run (extends base.yaml), spatial block split
experiments/      result JSONs and training logs (script output only)
scripts/          tables, figures, release packaging and download
paper/            LaTeX source of the IEEE paper, its tables and figures
docs/             protocol, contracts, compute, data card, model card, code style
```

The code follows the style of the course notebooks ([docs/code_style.md](docs/code_style.md)):
simple functions and a plain PyTorch training loop.

### Where to look in the code

| File | What it does |
|---|---|
| [src/hydroscope/models/factory.py](src/hydroscope/models/factory.py) | Builds the three models (RGB or 13 bands, ImageNet or SSL4EO-S12 weights) and picks the layers trained by `head` and `last_block` |
| [src/hydroscope/training/train.py](src/hydroscope/training/train.py) | EuroSAT training: `run_epoch` / `fit` loop, warmup + cosine, early stopping |
| [src/hydroscope/training/finetune.py](src/hydroscope/training/finetune.py) | Fine-tuning on Aguascalientes in `head` or `last_block` mode, with the class-weighted loss |
| [src/hydroscope/training/evaluate.py](src/hydroscope/training/evaluate.py) | Metrics with `sklearn.metrics` and the result JSON |
| [src/hydroscope/data/](src/hydroscope/data/) | EuroSAT and Aguascalientes datasets, labels, normalization, resize and augmentation |
| [src/hydroscope/explain/gradcam.py](src/hydroscope/explain/gradcam.py) | Grad-CAM maps |
| [configs/base.yaml](configs/base.yaml) + [tests/test_config.py](tests/test_config.py) | The shared protocol, and the tests that keep every run config on it |
| [scripts/make_results.py](scripts/make_results.py) | Paper tables, figures, text numbers and the results table below, from the result JSONs |

---

## How to run it

### Prerequisites

1. Download or clone this repository in your local system.
2. Install Miniconda or Anaconda if you do not have it.
3. Open a terminal (on Windows, the *Anaconda Prompt*) in the repository folder and create the
   environment with every library the project needs:

   ```bash
   conda env create -f environment.yml
   conda activate hydroscope
   python -m pytest -q
   ```

   The last command runs the tests. If they pass, the environment is ready.

The trained models and the labeled Aguascalientes patches are too large for the repository.
They are files of the release
[v1.0](https://github.com/alejandro-serrano/ags-hydroscope/releases/tag/v1.0).

Every time you open a new terminal, activate the environment again with
`conda activate hydroscope`.

### Reproduce the results

There are three levels. Start with the first one: it takes no downloads.

| Level | What you get | What it needs |
|---|---|---|
| 1. Tables and figures | Every table, result figure and number of the paper, from the committed results | Only this repository |
| 2. Evaluation | The Aguascalientes results, recomputed from the trained models | The release files |
| 3. Training | Everything, from scratch | EuroSAT, the release labels and time ([docs/compute.md](docs/compute.md)) |

**Level 1 · Tables and figures from the committed results**

```bash
python scripts/make_results.py
```

It reads the result files in `experiments/results/` and rewrites the paper's tables
(`paper/tables/`), its result figures (`paper/figures/`) and the [Results](#results) table of
this README. The values it writes should be the same as in the Results table below.

**Level 2 · Evaluate the released checkpoints on Aguascalientes**

First download the release files. This command puts the 24 trained models in
`checkpoints/` and the labeled patches in `data/ags/`, and checks that every file arrived
complete:

```bash
python scripts/download_release.py
```

You can also download them by hand from the
[release page](https://github.com/alejandro-serrano/ags-hydroscope/releases/tag/v1.0): the
`.pt` and `.yaml` files go in `checkpoints/`, and `ags_labeled_v1.0.zip` is unzipped in the
repository folder.

Then evaluate a model on the Aguascalientes test patches. This one is the selected model:

```bash
python -m hydroscope.training.evaluate --ckpt checkpoints/resnet50_ms13_ssl4eo_ft-last_block.pt --split ags_test
python scripts/make_results.py
```

Repeat the first command with any of the 24 files in `checkpoints/`: the eight trained on
EuroSAT give the zero-shot results, and the `_ft-head` and `_ft-last_block` ones give the
fine-tuned results. Each run rewrites its result file in `experiments/results/`.

**Level 3 · Retrain from scratch**

```bash
python scripts/prepare_eurosat.py
python -m hydroscope.training.train --config configs/resnet50_rgb_imagenet.yaml
python scripts/run_ags.py
python scripts/make_results.py
```

The first command downloads EuroSAT with TorchGeo. The second trains one model; repeat it with
each of the eight run files in `configs/` (all except `base.yaml` and `smoke.yaml`). The third
runs the 8 zero-shot evaluations and the 16 fine-tunings on Aguascalientes, and the last one
rebuilds the tables and figures.

The runs of the study used a Mac with an Apple M2 Pro and the MPS backend (Apple's GPU
support in PyTorch), in fp32. To check that training works before a long run, use the quick
version with 200 images and one epoch:
`python -m hydroscope.training.train --config configs/smoke.yaml`.

The labels are manual and come with the release. To rebuild the Aguascalientes composite,
grid and patches from Earth Engine, see `src/hydroscope/geo/` and
[docs/data_card.md](docs/data_card.md).

### If something fails

**`conda` is not recognized.** On Windows, use the *Anaconda Prompt* instead of the regular
terminal. On macOS or Linux, close the terminal and open a new one after installing Miniconda.

**`ModuleNotFoundError: No module named 'hydroscope'`.** The environment is not active. Run
`conda activate hydroscope` and try again.

**Your speed or memory values differ from ours.** That is expected: `patches_per_s` and
`peak_mem_mb` depend on the computer. Accuracy and F1 should be the same.

**`scripts/run_ags.py` finishes without running anything.** It skips every job that already
has a result file. To recompute them, first move the `*_ags_test.json` files out of
`experiments/results/`.

**The Grad-CAM or study-area figures are not rebuilt.** Level 1 does not rebuild them.
`scripts/make_gradcam.py` needs the release files of level 2 and EuroSAT
(`python scripts/prepare_eurosat.py`); `scripts/make_paper_inputs.py` also needs the
Aguascalientes composite, which comes from Earth Engine.

---

## Results

Generated by `scripts/make_results.py` from `paper/tables/summary.csv`; do not edit by hand.

<!-- results:start -->
| Model | Bands | Weights | EuroSAT macro-F1 | Ags zero-shot | Ags last_block |
| --- | --- | --- | ---: | ---: | ---: |
| ResNet-50 | RGB | ImageNet | 0.983 | 0.259 | 0.278 |
| ResNet-50 | MS13 | ImageNet | 0.985 | 0.271 | 0.371 |
| ResNet-50 | MS13 | SSL4EO | 0.993 | 0.354 | 0.525 |
| EfficientNet-B0 | RGB | ImageNet | 0.984 | 0.271 | 0.400 |
| EfficientNet-B0 | MS13 | ImageNet | 0.985 | 0.196 | 0.361 |
| ViT-S | RGB | ImageNet | 0.991 | 0.326 | 0.441 |
| ViT-S | MS13 | ImageNet | 0.990 | 0.287 | 0.361 |
| ViT-S | MS13 | SSL4EO | 0.987 | 0.287 | 0.428 |

Ags = Aguascalientes test split, macro-F1 over the 9 classes with at least 10 test patches; last_block = after fine-tuning the last block and the head.

Model chosen by the selection rule (protocol section 8): **ResNet-50 / MS13 / SSL4EO**, Ags last_block macro-F1 0.525.
<!-- results:end -->

Full tables, accuracy, per-class F1, drop and recovery are in the paper; the selection rule
is in [docs/protocol.md](docs/protocol.md) section 8.

---

## Design decisions

**One protocol for every model.** No hyperparameter was changed for a single model, so a
difference between models comes from the architecture and not from tuning. A test fails if a
run config overrides a shared key.

**A spatial split, not a random one.** Neighboring cells are almost identical. A random split
would put near-duplicates in the fine-tune and test sets and inflate the scores, so whole
blocks of 8×8 cells go to one side.

**Macro-F1 as the main metric.** HerbaceousVegetation is by far the largest class of the
Aguascalientes test set, so accuracy rewards a model that predicts it everywhere. Macro-F1
counts a rare class as much as a common one.

**The same processing level as EuroSAT.** The Aguascalientes composite is Sentinel-2 L1C, like
EuroSAT, so the main thing that changes between the two is the landscape.

**A selection rule fixed before any result.** The selected model is the one with the highest
Aguascalientes macro-F1 after `last_block`. Fixing the rule first keeps us from choosing the
criterion that favors a model after the fact.

**No number typed by hand.** The paper's tables, figures and in-text numbers, and the results
table above, are written by `scripts/make_results.py` from the result files.

---

## Data and licenses

| Source | Used for | License |
|---|---|---|
| [EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al.), loaded with [TorchGeo](https://github.com/microsoft/torchgeo) | Training, validation and test | MIT |
| Sentinel-2 L1C harmonized (`COPERNICUS/S2_HARMONIZED`) via Google Earth Engine | Aguascalientes composite and patches | Copernicus Sentinel data terms: contains modified Copernicus Sentinel-2 data (2024) |
| [FAO GAUL 2025](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2025_level1) level 1 | State boundary | CC BY 4.0 (FAO. 2025. Global Administrative Unit Layers (GAUL)) |
| [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) weights via TorchGeo (ResNet-50 MoCo, ViT-S DINO) | Self-supervised initialization | CC BY 4.0 |
| [timm](https://github.com/huggingface/pytorch-image-models) library | Model architectures | Apache-2.0 |
| ImageNet weights loaded with timm (`resnet50.a1_in1k`, `efficientnet_b0.ra_in1k`, `vit_small_patch16_224.augreg_in21k_ft_in1k`) | ImageNet initialization | Pretrained weights keep the terms of their sources (see each model card on Hugging Face) |
| Aguascalientes labels and block split (this project) | Fine-tuning and test | MIT, as the code |

---

## Credits

Final project of the *Deep Learning* module of the Master's in Artificial Intelligence and
Data Science, Universidad Panamericana · Campus Aguascalientes.

Author: **[Manuel Alejandro Serrano Macias](https://mx.linkedin.com/in/alejandro-serrano)**.

### Citation

```bibtex
@misc{serrano2026hydroscope,
  title        = {{CNNs} vs.\ Vision Transformers under Geographic Domain Shift:
                  From {EuroSAT} to Semi-Arid {Aguascalientes}, {Mexico}},
  author       = {Serrano Macias, Manuel Alejandro},
  year         = {2026},
  howpublished = {\url{https://github.com/alejandro-serrano/ags-hydroscope}},
  note         = {Deep Learning course project, Universidad Panamericana}
}
```

## License

MIT. It covers the code, the labels and the checkpoints of this repository. Third-party data
and weights keep their own licenses and attribution (see above); the checkpoints start from
the timm or SSL4EO-S12 weights.
