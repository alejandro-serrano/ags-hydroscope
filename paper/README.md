# Paper

LaTeX source of the IEEE conference paper of Hydroscope Aguascalientes. The built PDF,
[`main.pdf`](main.pdf), is committed, so you can read it without building anything.

## Build the PDF

### With Tectonic (what `make paper` uses)

From the repository root:

```bash
make paper                  # same as: tectonic paper/main.tex
```

It writes `paper/main.pdf`. Tectonic runs LaTeX and BibTeX as many times as needed and downloads
the LaTeX packages it is missing, so the first build needs an internet connection.

To install Tectonic:

- macOS: `brew install tectonic`
- Any system with conda: `conda install -c conda-forge tectonic` (it can go in the `hydroscope`
  environment)
- Other options (installer script, prebuilt binaries): <https://github.com/tectonic-typesetting/tectonic>

### With TeX Live or MacTeX

From the repository root:

```bash
cd paper
latexmk -pdf main.tex       # writes main.pdf
latexmk -c                  # optional: deletes most auxiliary files
```

Without `latexmk`, run the four steps by hand:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

MacTeX and a full TeX Live already include everything the paper uses. On Debian or Ubuntu with a
smaller TeX Live, install these packages first (the IEEEtran class is in `texlive-publishers`):

```bash
sudo apt install texlive-latex-extra texlive-science texlive-publishers texlive-fonts-recommended latexmk
```

The auxiliary files (`*.aux`, `*.log`, `*.bbl`, `*.blg`) are ignored by git.

## What the PDF is built from

| Path | Content | Written by |
|---|---|---|
| `main.tex` | Preamble, title, abstract and order of the sections | By hand |
| `sections/*.tex` | The seven sections, from Introduction to Conclusions | By hand |
| `references.bib` | The five references | By hand |
| `tables/eurosat.tex`, `ags.tex`, `ags_accuracy.tex`, `cost.tex`, `macros.tex`, `summary.csv` | Result tables and every number quoted in the text | `scripts/make_results.py` |
| `figures/convergence_*.pdf`, `drop_recovery.pdf`, `ags_f1_per_class.pdf`, `confusion_selected.pdf` | Result figures | `scripts/make_results.py` |
| `tables/config.tex`, `tables/ags_classes.tex`, `figures/study_area.pdf` | Shared configuration, labeled patches per class and the study area | `scripts/make_paper_inputs.py` |
| `figures/gradcam_*.pdf`, `figures/gradcam_samples.csv` | Grad-CAM figures and the list of their patches | `scripts/make_gradcam.py` |

Do not edit the generated files by hand. If the results change, run
`python scripts/make_results.py` from the repository root and build the PDF again: the tables, the
figures and the numbers in the text are updated together. `make_gradcam.py` needs the release
files and EuroSAT, and `make_paper_inputs.py` also needs the Aguascalientes composite from Earth
Engine (see [If something fails](../README.md#if-something-fails) in the main README).
