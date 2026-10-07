# Literature

Key references for the paper. Every entry in `paper/references.bib` was fetched from its DOI (see
[How references.bib was built](#how-referencesbib-was-built)); none was written by hand.

"Main reported result" quotes only values found in the named source. "Check PDF" means the value is
not in the abstract and must be read from the full text before it is used in the paper.

| Key | Models | Data | Main reported result (source) | Finding relevant to our project | Cited in |
| --- | --- | --- | --- | --- | --- |
| `helber2019eurosat` | Deep CNNs (specific architectures: check PDF) | EuroSAT: Sentinel-2, 13 spectral bands, 10 classes, 27,000 labeled and geo-referenced images | Overall classification accuracy of 98.57 % (abstract) | Defines the source dataset, classes and bands we reuse, and the CNN benchmark that E1 compares against | Introduction, Related Work, Data |
| `bazi2021vit` | Vision Transformer; compressed version with half of the multihead attention layers removed | Merced, AID, Optimal31, NWPU | Average accuracy 98.49 %, 95.86 %, 95.56 % and 93.83 % on Merced, AID, Optimal31 and NWPU; compressed version 97.90 %, 94.27 %, 95.30 % and 93.05 % (abstract) | ViTs are competitive with CNN state of the art for remote-sensing scene classification; the datasets listed are not Sentinel-2 multispectral and the abstract reports no geographic domain shift (check PDF) | Introduction, Related Work |
| `wang2023ssl4eo` | Self-supervised pre-training: MoCo, DINO, MAE, data2vec | SSL4EO-S12: unlabeled, large-scale, global, multimodal (Sentinel-1/2), multiseasonal satellite imagery | No numbers in the abstract; it reports that benchmark results show SSL4EO-S12's effectiveness compared to existing datasets (numeric results: check PDF) | Source of the SSL weights in E2 (ResNet-50 MoCo, ViT-S/16 DINO, 13 bands) and the motivation for RQ-e | Related Work, Methodology |

## Sources

- `bazi2021vit`: abstract from the Crossref record of the DOI (`https://api.crossref.org/works/<DOI>`).
- `helber2019eurosat`, `wang2023ssl4eo`: Crossref has no abstract for these IEEE DOIs; the abstract
  was taken from the Semantic Scholar record of the DOI
  (`https://api.semanticscholar.org/graph/v1/paper/DOI:<DOI>?fields=title,abstract`).

## How references.bib was built

```bash
for d in 10.1109/JSTARS.2019.2918242 10.3390/rs13030516 10.1109/MGRS.2023.3281651; do
  curl -sLH "Accept: application/x-bibtex" "https://doi.org/$d"
done
```

The following mechanical normalizations were then applied with `sed` (no other field was edited):

- Keys renamed: `Helber_2019` → `helber2019eurosat`, `Bazi_2021` → `bazi2021vit`,
  `Wang_2023` → `wang2023ssl4eo`.
- Months converted to BibTeX macros (`July` → `jul`, `Feb` → `feb`, `Sept` → `sep`); `July` and
  `Sept` are undefined strings in BibTeX.
- Unicode en dash in `pages` converted to `--`; the T1 Times font has no glyph for U+2013.
- Proper nouns in titles braced (`{EuroSAT}`, `{SSL4EO-S12}`, `{Earth}`) because IEEEtran.bst
  lowercases titles.

## Known metadata issues (not fixed)

The DOI metadata splits some Arabic surnames, so IEEEtran renders them as "M. M. A. Rahhal",
"R. A. Dayil", "N. A. Ajlan" (`bazi2021vit`) and "N. A. A. Braham" (`wang2023ssl4eo`). Correcting
them would require editing the author field by hand; verify against the published PDFs before the
final submission.
