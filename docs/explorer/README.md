# Hydroscope Explorer

An optional, interactive companion to the paper. It shows the study's data and results in
eight demos:

1. **EuroSAT vs. Aguascalientes**: random sample patches of every class from both datasets, in
   true color.
2. **Results on EuroSAT**: the training curves of every run (validation macro-F1 and accuracy
   per epoch) and its score on the EuroSAT test set.
3. **Drop and recovery**: macro-F1 on EuroSAT, zero-shot on Aguascalientes and after each
   fine-tuning mode, for every run.
4. **Confusion matrix**: the confusion matrix of any run, on EuroSAT or on Aguascalientes
   (zero-shot and after each fine-tuning mode).
5. **CNN vs. ViT**: accuracy against cost (throughput, parameters, time per epoch, peak
   memory) for the two CNNs and the Vision Transformer.
6. **Grad-CAM**: the Grad-CAM maps of each model on the same EuroSAT and Aguascalientes patches,
   with an adjustable opacity.
7. **Spatial split**: the fine-tune and test cells of the 8×8-cell block split.
8. **Code tour**: the functions behind each step of the study, from the Earth Engine composite
   to the selection rule.

## How to open it

Double-click `index.html`. It works without internet and without a server: every script,
style and font is in this folder. Only the links to GitHub (the repository and the source
lines of the code tour) need a connection.

## Which files are generated

| File | Written by |
| --- | --- |
| `data.js` | `scripts/make_explorer_data.py`, which also runs at the end of `scripts/make_results.py`. It reads the result JSONs of `experiments/results/`, the EuroSAT training logs of `experiments/logs/` (the curves) and `configs/base.yaml` (the shared training settings). Do not edit by hand. |
| `basemap.jpg`, `basemap.js` | `python scripts/make_explorer_basemap.py` (run it by hand; `make_results.py` does not call it). Do not edit by hand. |
| `samples.png`, `gradcam.png`, `images.js` | `python scripts/make_explorer_images.py` (run it by hand; `make_results.py` does not call it). `samples.png` holds 6 random patches per class (seed 0) of the EuroSAT test set and of the labeled Aguascalientes patches; `gradcam.png` holds the patches and Grad-CAM maps of the paper's figures, read from `paper/figures/gradcam_samples.csv`; `images.js` says where each 64×64 tile sits. It needs the EuroSAT cache, `data/ags/labels.csv` with its patches and the checkpoints; without them the matching part of `images.js` is null. Do not edit by hand. |
| `index.html`, `explorer.css`, `explorer.js` | By hand. |
| `vendor/` | Third-party copies (see below). |

After new evaluations, run `python scripts/make_results.py` and reload the page.

## Background image of the spatial split

`basemap.jpg` is a mosaic of the cell previews in `data/ags/previews/`: each 256×256 preview
is reduced to 8×8 pixels and placed at its grid row and column, on a white background.
`basemap.js` tells the page where the mosaic starts (`row0`, `col0`) and its size in cells.
The image is optional: the page works without it.

The previews are the RGB images (B04, B03, B02, with the global 2–98 % stretch of the
composite) that were used for labeling. They are **not** in git and **not** in release v1.0,
whose data zip holds only `labels.csv`, `configs/ags_block_split.csv`, the labeled patches
(`.tif`) and `NOTICE`. They are written by `python -m hydroscope.geo.tiling` from the Earth
Engine composite (see [docs/data_card.md](../data_card.md)). Without them,
`make_explorer_basemap.py` prints a note and leaves the existing files unchanged.

Contains modified Copernicus Sentinel-2 data (2024).

## Without `data/ags/labels.csv`

`labels.csv` is not in git, so `data.js` may have no split data. The spatial split demo then
explains how to get the file (it comes with the release) and how to rebuild `data.js`. The
other three demos work without it.

## Third-party licenses

| File | License |
| --- | --- |
| `samples.png`, `gradcam.png`: EuroSAT patches ([EuroSAT](https://github.com/phelber/EuroSAT), Helber et al.) | MIT |
| `samples.png`, `gradcam.png`, `basemap.jpg`: Aguascalientes patches | Contains modified Copernicus Sentinel-2 data (2024) |
| `vendor/d3.v7.9.0.min.js` ([D3](https://d3js.org)) | ISC, `vendor/LICENSE-d3.txt` |
| `vendor/prism-core.min.js`, `vendor/prism-python.min.js` ([Prism](https://prismjs.com)) | MIT, `vendor/LICENSE-prism.txt` |
| `vendor/fonts/sora-*.woff2` (Sora) | SIL OFL 1.1, `vendor/fonts/LICENSE-sora.txt` |
| `vendor/fonts/inter-*.woff2` (Inter) | SIL OFL 1.1, `vendor/fonts/LICENSE-inter.txt` |
| `vendor/fonts/ibm-plex-mono-*.woff2` (IBM Plex Mono) | SIL OFL 1.1, `vendor/fonts/LICENSE-ibm-plex-mono.txt` |
