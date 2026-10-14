# Hydroscope Explorer

An optional, interactive companion to the paper. It shows the same results as the paper's
tables and figures in four demos:

1. **Drop and recovery**: macro-F1 on EuroSAT, zero-shot on Aguascalientes and after each
   fine-tuning mode, for every run.
2. **Confusion matrix**: the confusion matrix of any run, on EuroSAT or on Aguascalientes
   (zero-shot and after each fine-tuning mode).
3. **Spatial split**: the fine-tune and test cells of the 8×8-cell block split.
4. **Code tour**: the functions behind each step of the study, from the Earth Engine composite
   to the selection rule.

## How to open it

Double-click `index.html`. It works without internet and without a server: every script,
style and font is in this folder. Only the links to GitHub (the repository and the source
lines of the code tour) need a connection.

## Which files are generated

| File | Written by |
| --- | --- |
| `data.js` | `scripts/make_explorer_data.py`, which also runs at the end of `scripts/make_results.py`. Do not edit by hand. |
| `index.html`, `explorer.css`, `explorer.js` | By hand. |
| `vendor/` | Third-party copies (see below). |

After new evaluations, run `python scripts/make_results.py` and reload the page.

## Without `data/ags/labels.csv`

`labels.csv` is not in git, so `data.js` may have no split data. The spatial split demo then
explains how to get the file (it comes with the release) and how to rebuild `data.js`. The
other three demos work without it.

## Third-party licenses

| File | License |
| --- | --- |
| `vendor/d3.v7.9.0.min.js` ([D3](https://d3js.org)) | ISC, `vendor/LICENSE-d3.txt` |
| `vendor/prism-core.min.js`, `vendor/prism-python.min.js` ([Prism](https://prismjs.com)) | MIT, `vendor/LICENSE-prism.txt` |
| `vendor/fonts/sora-*.woff2` (Sora) | SIL OFL 1.1, `vendor/fonts/LICENSE-sora.txt` |
| `vendor/fonts/inter-*.woff2` (Inter) | SIL OFL 1.1, `vendor/fonts/LICENSE-inter.txt` |
| `vendor/fonts/ibm-plex-mono-*.woff2` (IBM Plex Mono) | SIL OFL 1.1, `vendor/fonts/LICENSE-ibm-plex-mono.txt` |
