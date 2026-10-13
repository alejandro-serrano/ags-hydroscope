# Literature

Key references for the paper. Every entry in `paper/references.bib` was fetched from its DOI or from
DBLP (see [How references.bib was built](#how-referencesbib-was-built)). The only exception is
`inagua2022planhidrico`, written by hand from the PDF (see
[Exception: hand-written entry](#exception-hand-written-entry)).

"Main reported result" quotes only values found in the named source. "Check PDF" means the value is
not in the abstract and must be read from the full text before it is used in the paper.

| Key | Models | Data | Main reported result (source) | Finding relevant to our project | Cited in |
| --- | --- | --- | --- | --- | --- |
| `helber2019eurosat` | Deep CNNs (specific architectures: check PDF) | EuroSAT: Sentinel-2, 13 spectral bands, 10 classes, 27,000 labeled and geo-referenced images | Overall classification accuracy of 98.57 % (abstract) | Defines the source dataset, classes and bands we reuse, and the CNN benchmark that E1 compares against | Related Work, Data |
| `bazi2021vit` | Vision Transformer; compressed version with half of the multihead attention layers removed | Merced, AID, Optimal31, NWPU | Average accuracy 98.49 %, 95.86 %, 95.56 % and 93.83 % on Merced, AID, Optimal31 and NWPU; compressed version 97.90 %, 94.27 %, 95.30 % and 93.05 % (abstract) | ViTs are competitive with CNN state of the art for remote-sensing scene classification; the datasets listed are not Sentinel-2 multispectral and the abstract reports no geographic domain shift (check PDF) | Related Work, Discussion |
| `wang2023ssl4eo` | Self-supervised pre-training: MoCo, DINO, MAE, data2vec | SSL4EO-S12: unlabeled, large-scale, global, multimodal (Sentinel-1/2), multiseasonal satellite imagery | No numbers in the abstract; it reports that benchmark results show SSL4EO-S12's effectiveness compared to existing datasets (numeric results: check PDF) | Source of the SSL weights in E2 (ResNet-50 MoCo, ViT-S/16 DINO, 13 bands) and the motivation for RQ-e | Related Work, Methodology, Discussion |
| `sumbul2019bigearthnet` | Shallow CNN vs. a CNN pre-trained on ImageNet | BigEarthNet: 590,326 Sentinel-2 patches, multi-label (CORINE Land Cover 2018) | No numbers in the abstract; a shallow CNN trained on BigEarthNet "provides much higher accuracy" than an ImageNet-pre-trained CNN (abstract) | Second Sentinel-2 benchmark introduced with CNNs; first hint that ImageNet features do not fully transfer | Related Work |
| `he2016resnet` | ResNet (residual networks, up to 152 layers) | ImageNet, CIFAR-10, COCO | 3.57 % error on the ImageNet test set with an ensemble (abstract) | Backbone of our ResNet-50 runs | Related Work, Methodology |
| `tan2019efficientnet` | EfficientNet family (compound scaling) | ImageNet, CIFAR-100, Flowers and 3 other transfer datasets | EfficientNet-B7: 84.3 % top-1 on ImageNet (abstract, arXiv) | Backbone of our EfficientNet-B0 runs | Related Work, Methodology |
| `dosovitskiy2020vit` | Vision Transformer (ViT) | ImageNet, CIFAR-100, VTAB | No numbers in the abstract; "excellent results compared to state-of-the-art convolutional networks" (abstract, arXiv) | Backbone of our ViT-S/16 runs | Related Work, Methodology |
| `cong2022satmae` | SatMAE: masked autoencoder pre-training of transformers | Temporal and multi-spectral satellite imagery | Up to 7 % (supervised) and up to 14 % (land-cover transfer) improvement over previous methods (abstract, arXiv) | Transformers can use multispectral bands when pre-training is designed for them; context for RQ-d | Related Work |
| `wang2022sslreview` | Review of self-supervised learning (SSL) for remote sensing | Popular remote-sensing datasets (preliminary benchmark) | No numbers in the abstract | Frames SSL as the answer to scarce labels; context for RQ-e | Related Work |
| `manas2021seco` | Seasonal Contrast (SeCo), self-supervised pre-training | Unlabeled, uncurated Sentinel-2 images of many locations and dates | No numbers in the abstract; SeCo models beat ImageNet pre-trained counterparts on multiple downstream tasks (abstract) | Same motivation as ours: ImageNet features may not generalize because of the domain gap | Related Work |
| `tuia2016domain` | Review of domain adaptation (DA) methods | Remote sensing images (very high spatial and spectral resolution) | Review; no numbers in the abstract | Defines the spectral shift between training and mapping regions that our study measures | Related Work |
| `koh2021wilds` | Benchmark of distribution shifts | 10 datasets, including satellite imaging and poverty mapping across time and location | No numbers in the abstract; out-of-distribution performance substantially lower than in-distribution (abstract, arXiv) | Shows that location shift in satellite imaging is a known, general problem | Related Work |
| `roberts2017cv` | Cross-validation strategies (review and simulations) | Ecological data with temporal, spatial, hierarchical or phylogenetic structure | No numbers in the abstract; block cross-validation "nearly universally more appropriate than random cross-validation" for prediction to new data (abstract) | Justifies the spatial block split of Aguascalientes | Related Work, Data |
| `drusch2012sentinel2` | Sentinel-2 mission description | Sentinel-2 | Abstract not available from Crossref, Semantic Scholar or OpenAlex; only the title is used | Source for the Sentinel-2 mission | Data |
| `gorelick2017gee` | Google Earth Engine platform | Planetary-scale geospatial data | No numbers in the abstract | Tool used to build the April 2024 composite | Data |
| `stewart2022torchgeo` | TorchGeo library | Benchmark datasets and pre-trained multispectral models | No numbers in the abstract | Tool that provides EuroSAT, its split and the SSL4EO-S12 weights | Data |
| `deng2009imagenet` | ImageNet database | 3.2 million images in 5247 synsets (state at publication) | No accuracy reported (abstract, OpenAlex) | Source of the ImageNet weights | Methodology |
| `inagua2022planhidrico` | State water plan (no models) | Aguascalientes water resources | Not used: no figure from the plan appears in the paper | Motivation: groundwater is the main and overexploited source; demand is mainly agricultural and urban public use | Introduction |

## Claims and supporting text

Every sentence of the paper about a reference, with the text that supports it. Abstract quotes are
verbatim; the plan quotes are verbatim Spanish from the PDF. The plan's printed page numbers start
at its Introduction (printed p. 1 = PDF p. 13).

| Key | Claim in the paper (section) | Supporting text (source) |
| --- | --- | --- |
| `inagua2022planhidrico` | Groundwater, the main source of supply, is overexploited, and the plan names this as the central problem of the state (Introduction) | "enfrentan un problema central con el que ha convivido el estado de Aguascalientes desde la década de los setenta: La sobreexplotación de los recursos hídricos subterráneos que constituyen su principal fuente de abastecimiento." (printed p. 2, PDF p. 14); also "se siguen sobreexplotando los acuíferos" (printed p. 3, PDF p. 15) |
| `inagua2022planhidrico` | Demand for water comes mainly from agricultural and urban public use (Introduction) | "Existe un desequilibrio entre la demanda de agua, principalmente generada por los usos agrícola y público urbano, y la capacidad natural de renovación del agua en cuencas y acuíferos" (printed p. 2, PDF p. 14) |
| `helber2019eurosat` | EuroSAT was introduced with a benchmark of deep CNNs over 13 bands and 10 classes, and they reached a high overall accuracy (Related Work) | "We present a novel dataset, based on these images that covers 13 spectral bands and is comprised of ten classes"; "Benchmarks are provided for this novel dataset with its spectral bands using state-of-the-art deep convolutional neural networks. An overall classification accuracy of 98.57% was achieved" (abstract, Semantic Scholar) |
| `sumbul2019bigearthnet` | Much larger Sentinel-2 archive with several land-cover labels per patch (Related Work) | "The BigEarthNet is significantly larger than the existing archives in remote sensing"; "each image patch is annotated by multiple land-cover classes (i.e., multi-labels)" (abstract, Semantic Scholar) |
| `sumbul2019bigearthnet` | A shallow CNN trained on it gave much higher accuracy than a CNN pre-trained on ImageNet (Related Work) | "a shallow Convolutional Neural Network (CNN) architecture trained on the BigEarthNet provides much higher accuracy compared to a state-of-the-art CNN model pre-trained on the ImageNet" (abstract, Semantic Scholar) |
| `he2016resnet` | ResNet learns residual functions, which makes very deep networks easier to optimize (Related Work) | "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs"; "these residual networks are easier to optimize" (abstract, Semantic Scholar) |
| `tan2019efficientnet` | Scales depth, width and resolution with one compound coefficient; better accuracy and efficiency than previous CNNs (Related Work) | "uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient"; "EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets" (abstract, arXiv 1905.11946) |
| `dosovitskiy2020vit` | Pure transformer on a sequence of image patches, without convolutions; excellent results vs. CNNs with fewer resources when pre-trained on large data (Related Work) | "this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well"; "When pre-trained on large amounts of data ... Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train." (abstract, arXiv 2010.11929) |
| `bazi2021vit` | ViT applied to remote-sensing scene classification; attention derives long-range relations between pixels; promising results on four datasets (Related Work) | "we propose a remote-sensing scene-classification method based on vision transformers"; "they use multihead attention mechanisms as the main building block to derive long-range contextual relation between pixels in images"; "demonstrate the promising capability of the model compared to state-of-the-art methods"; four datasets: "Merced, AID, Optimal31 and NWPU" (abstract, Crossref) |
| `cong2022satmae` | Masked-autoencoder pre-training of transformers for temporal or multispectral satellite images; band groups with own spectral position encodings are beneficial; transfer gains including land-cover classification (Related Work) | "we present SatMAE, a pre-training framework for temporal or multi-spectral satellite imagery based on Masked Autoencoder (MAE)"; "encoding multi-spectral data as groups of bands with distinct spectral positional encodings is beneficial"; "transfer learning performance on downstream remote sensing tasks, including land cover classification" (abstract, arXiv 2207.08051) |
| `wang2022sslreview` | Introduces SSL for remote sensing; notes most of its potential in Earth observation remains locked (Related Work) | "we provide an introduction to and a review of the concepts and latest developments in SSL for computer vision in the context of remote sensing"; "most of the potential of SSL in the domain of Earth observation remains locked" (abstract, Semantic Scholar) |
| `manas2021seco` | ImageNet models may not generalize to remote sensing because of the domain gap; gathers unlabeled images of many locations and times; invariance to time and position; beat ImageNet counterparts on several downstream tasks (Related Work) | "most of these methods are pre-trained on ImageNet and their generalization to remote sensing imagery is not guaranteed due to the domain gap"; "datasets containing images from multiple Earth locations at different timestamps"; "takes advantage of time and position invariance to learn transferable representations"; "models trained with SeCo achieve better performance than their ImageNet pre-trained counterparts ... on multiple downstream tasks" (abstract, Semantic Scholar) |
| `wang2023ssl4eo` | Large, global, multiseasonal Sentinel-1/2 corpus with pre-trained models for methods such as MoCo and DINO (Related Work) | "a large-scale, global, multimodal, and multiseasonal corpus of satellite imagery"; "a set of representative methods: momentum contrast (MoCo), self-distillation with no labels (DINO), masked autoencoders (MAE), and data2vec"; "The dataset, related source code, and pretrained models are available" (abstract, Semantic Scholar) |
| `tuia2016domain` | Spectral shifts between training and mapping regions are likely to make the model fail; caused by acquisition and atmospheric conditions or by changes in the observed objects; reviews DA methods robust to such shifts (Related Work) | "When training samples are collected from an image or a spatial region that is different from the one used for mapping, spectral shifts between the two distributions are likely to make the model fail. Such shifts are generally due to differences in acquisition and atmospheric conditions or to changes in the nature of the object observed. To design classification methods that are robust to data set shifts, recent remote sensing literature has considered solutions based on domain adaptation (DA) approaches." (abstract, Semantic Scholar) |
| `koh2021wilds` | Collects real distribution shifts, including across time and location in satellite imaging; much lower performance out of distribution (Related Work) | "a curated benchmark of 10 datasets reflecting a diverse range of distribution shifts that naturally arise in real-world applications, such as ... across time and location in satellite imaging"; "standard training yields substantially lower out-of-distribution than in-distribution performance" (abstract, arXiv 2012.07421) |
| `roberts2017cv` | Random cross-validation on structured data can seriously underestimate prediction error; recommend block cross-validation (Related Work, Data) | "when performing cross-validation, these structures are regularly ignored, resulting in serious underestimation of predictive error"; "Block cross-validation, where data are split strategically rather than randomly, can address these issues"; "We recommend that block cross-validation be used wherever dependence structures exist in a dataset" (abstract, Crossref) |
| `drusch2012sentinel2` | Cited for the Sentinel-2 mission only (Data) | Title: "Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services" (no abstract available) |
| `gorelick2017gee` | Cited for Google Earth Engine only (Data) | "Google Earth Engine is a cloud-based platform for planetary-scale geospatial analysis" (abstract, Semantic Scholar) |
| `stewart2022torchgeo` | Cited for TorchGeo only (Data) | "we introduce TorchGeo, a Python library for integrating geospatial data into the PyTorch deep learning ecosystem. TorchGeo provides data loaders for a variety of benchmark datasets" (abstract, Semantic Scholar) |
| `deng2009imagenet` | Cited for ImageNet only (Methodology) | "We introduce here a new database called "ImageNet", a large-scale ontology of images built upon the backbone of the WordNet structure." (abstract, OpenAlex) |

## Sources

- `bazi2021vit`: abstract from the Crossref record of the DOI (`https://api.crossref.org/works/<DOI>`).
- `helber2019eurosat`, `wang2023ssl4eo`: Crossref has no abstract for these IEEE DOIs; the abstract
  was taken from the Semantic Scholar record of the DOI
  (`https://api.semanticscholar.org/graph/v1/paper/DOI:<DOI>?fields=title,abstract`).
- `roberts2017cv`: abstract from the Crossref record of the DOI.
- `sumbul2019bigearthnet`, `tuia2016domain`, `wang2022sslreview`, `manas2021seco`, `gorelick2017gee`,
  `stewart2022torchgeo`, `he2016resnet`: Crossref has no abstract; taken from Semantic Scholar.
- `deng2009imagenet`: no abstract in Crossref or Semantic Scholar; taken from OpenAlex
  (`https://api.openalex.org/works/doi:<DOI>`, `abstract_inverted_index`).
- `drusch2012sentinel2`: no abstract in Crossref, Semantic Scholar (elided by the publisher) or
  OpenAlex; only its title is used.
- `tan2019efficientnet`, `dosovitskiy2020vit`, `koh2021wilds`, `cong2022satmae`: abstract from the
  arXiv record (`https://export.arxiv.org/api/query?id_list=<id>`; ids 1905.11946, 2010.11929,
  2012.07421, 2207.08051). The BibTeX is the conference version from DBLP.
- `inagua2022planhidrico`: the PDF itself (see below).

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

### Method references (F6)

```bash
for d in 10.1109/CVPR.2016.90 10.1109/ICCV.2017.74 \
         10.48550/arXiv.1905.11946 10.48550/arXiv.2010.11929; do
  curl -sLH "Accept: application/x-bibtex" "https://doi.org/$d"
done
```

The two arXiv DOIs return DataCite records. Same mechanical normalizations as above, plus one:

- Keys renamed: `He_2016` → `he2016resnet`, `Selvaraju_2017` → `selvaraju2017`,
  `https://doi.org/10.48550/arxiv.1905.11946` → `tan2019efficientnet`,
  `https://doi.org/10.48550/arxiv.2010.11929` → `dosovitskiy2020vit`.
- Months `June` → `jun`, `Oct` → `oct`; en dash in `pages` → `--`.
- Titles braced: `{Grad-CAM}`, `{EfficientNet}`.
- Entry type of `tan2019efficientnet` changed from `@article` to `@misc`: the DataCite record has
  no journal, so `@article` makes BibTeX warn; `@misc` is the type DataCite gives the ViT preprint.

### Related Work and Introduction references (F6-01)

```bash
for d in 10.1109/IGARSS.2019.8900532 10.1109/MGRS.2016.2548504 10.1111/ecog.02881 \
         10.1109/MGRS.2022.3198244 10.1109/ICCV48922.2021.00928 10.1016/j.rse.2011.11.026 \
         10.1016/j.rse.2017.06.031 10.1145/3557915.3560953 10.1109/CVPR.2009.5206848; do
  curl -sLH "Accept: application/x-bibtex" "https://doi.org/$d"
done
```

Conference papers come from DBLP. dblp.org answers `curl` with an anti-bot page, so each record
was opened in a browser and its BibTeX export copied verbatim:

- `https://dblp.org/rec/conf/icml/TanL19.html?view=bibtex` → `tan2019efficientnet` (ICML 2019)
- `https://dblp.org/rec/conf/iclr/DosovitskiyB0WZ21.html?view=bibtex` → `dosovitskiy2020vit` (ICLR 2021)
- `https://dblp.org/rec/conf/icml/KohSMXZBHYPGLDS21.html?view=bibtex` → `koh2021wilds` (ICML 2021)
- `https://dblp.org/rec/conf/nips/CongKMLRHBLE22.html?view=bibtex` → `cong2022satmae` (NeurIPS 2022)

The first two replace the arXiv `@misc` records of F6 (keys kept), so the `@misc` type change
described above no longer applies. Every title was checked against the requested paper. Same
mechanical normalizations as above, plus:

- Keys renamed: `Sumbul_2019` → `sumbul2019bigearthnet`, `Tuia_2016` → `tuia2016domain`,
  `Roberts_2017` → `roberts2017cv`, `Wang_2022` → `wang2022sslreview`, `Manas_2021` →
  `manas2021seco`, `Drusch_2012` → `drusch2012sentinel2`, `Gorelick_2017` → `gorelick2017gee`,
  `Stewart_2022` → `stewart2022torchgeo`, `Deng_2009` → `deng2009imagenet`, and the
  `DBLP:conf/...` keys to the four keys listed above.
- Months `July` → `jul`, `June` → `jun`, `Mar` → `mar`, `Dec` → `dec`, `Oct` → `oct`, `May` → `may`,
  `Nov` → `nov`; en dash in `pages` → `--`.
- Unicode hyphen U+2010 (`roberts2017cv` title and two surnames) → `-`; right quote U+2019
  (`drusch2012sentinel2` title, `stewart2022torchgeo` series) → `'`.
- Titles braced: `{ImageNet}`, `{Sentinel-2}`, `{ESA}`, `{GMES}`, `{Google Earth Engine}`,
  `{TorchGeo}`, `{EfficientNet}`, `{SatMAE}`.
- Capitalization fix: Crossref gives the BigEarthNet title as `Bigearthnet`; it was changed to
  `{BigEarthNet}`, the name used in the abstract.
- `\_` → `_` in the `url` of `cong2022satmae`: DBLP escapes the underscore, and IEEEtran prints
  URLs verbatim, so the escaped form showed a backslash in the PDF.

### Exception: hand-written entry

`inagua2022planhidrico` has no DOI and no BibTeX record. It was written by hand from the PDF:
Plan Hídrico Estatal 2021-2050 del Estado de Aguascalientes,
<https://eservicios2.aguascalientes.gob.mx/NormatecaAdministrador/archivos/EDO-23-180.pdf>
(230 pages, downloaded 2026-10-13; not stored in the repo).

- Title: cover, PDF p. 1 ("PLAN HÍDRICO ESTATAL 2021-2050 DEL ESTADO DE AGUASCALIENTES"); braced
  as a whole because it is a proper name in Spanish.
- `howpublished`, `month`, `year`: cover, PDF p. 1 ("Plan Publicado en la Edición Extraordinaria del
  Periódico Oficial del Estado de Aguascalientes, el martes 19 de julio de 2022.").
- Author: the Instituto del Agua del Estado prepares and proposes the plan (PDF p. 3: "será
  competencia del Instituto del Agua del Estado, elaborar, modificar y actualizar la propuesta del
  Plan Hidráulico"; "PROPONE: ... DIRECTOR GENERAL DEL INSTITUTO DEL AGUA DEL ESTADO"); the
  acronym INAGUA is the one the plan uses (e.g., table of contents, section 5.1.2).
- `note = {In Spanish}`.

The two statements cited in the Introduction are on printed p. 2 (PDF p. 14), with the
continued overexploitation repeated on printed p. 3; see the quotes above. No figure from the
plan is used in the paper.

## Known metadata issues (not fixed)

The DOI metadata splits some Arabic surnames, so IEEEtran renders them as "M. M. A. Rahhal",
"R. A. Dayil", "N. A. Ajlan" (`bazi2021vit`) and "N. A. A. Braham" (`wang2023ssl4eo`). Correcting
them would require editing the author field by hand; verify against the published PDFs before the
final submission.

Also from the fetched metadata (F6-01): Crossref drops the accents of `manas2021seco` (rendered
"O. Manas", "X. Giro-i Nieto", "D. Vazquez", "P. Rodriguez") and splits the surname of
`stewart2022torchgeo` (rendered "J. M. L. Ferres" for Lavista Ferres). Same policy: verify against
the published PDFs before the final submission.
