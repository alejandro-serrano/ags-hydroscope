# Hydroscope Aguascalientes

*Aguascalientes Water Monitor* — a comparative study of CNNs vs. a Vision Transformer for
Sentinel-2 land-use classification, pre-trained on EuroSAT and evaluated on Aguascalientes, Mexico
(semi-arid domain shift).

[![CI](https://github.com/alejandro-serrano/ags-hydroscope/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

## Overview

- **Models:** ResNet-50, EfficientNet-B0, ViT-S/16 (timm), with RGB or 13-band input.
- **Source domain:** EuroSAT (TorchGeo, official split, 10 classes).
- **Target domain:** Aguascalientes, Sentinel-2 L1C harmonized, April 2024 median, 640 m cells,
  spatial split by 5×5 km blocks.
- **Questions:** in-domain accuracy, zero-shot generalization, recovery after fine-tuning
  (`head`, `last_block`), computational cost, and Grad-CAM explanations.
- All models share one protocol ([docs/protocol.md](docs/protocol.md)); data formats are in
  [docs/contracts.md](docs/contracts.md).

## Setup

```bash
conda env create -f environment.yml
conda activate hydroscope
make test     # python -m pytest -q
make smoke    # CPU, 200 images, 1 epoch
```

Data and checkpoints are published as GitHub Release assets, not in the repository.
<!-- TODO: download instructions once the first release exists. -->

## Reproduce

```bash
make train CONFIG=configs/<model>_<bands>_<init>.yaml        # EuroSAT training
make eval  CKPT=checkpoints/<run_id>.pt SPLIT=ags_test        # zero-shot on Aguascalientes
python -m hydroscope.training.finetune --ckpt checkpoints/<run_id>.pt --mode head|last_block
make tables figures                                            # paper tables and figures
```

## Results

<!-- Generated from experiments/results/*.json by scripts/; do not type numbers here. -->
Results will be added once the runs are frozen.

## Repository layout

```
src/hydroscope/   data, models, training, explain, geo
configs/          one YAML per run (extends base.yaml)
experiments/      result JSONs (script output only)
paper/  slides/   IEEE paper and presentation
docs/             protocol, contracts, data card, model card
```

## Citation

```bibtex
@misc{serrano2026hydroscope,
  title  = {Hydroscope Aguascalientes: CNNs vs. Vision Transformers for Sentinel-2
            Land-Use Classification under Semi-Arid Domain Shift},
  author = {Serrano, Alejandro},
  year   = {2026},
  note   = {Deep Learning course project}
}
```

## License

[MIT](LICENSE). EuroSAT and Sentinel-2 data are subject to their own licenses.
