# Labeling guide (v2, 2026-10-09)

Spanish translation: [`GUIDE.es.md`](GUIDE.es.md). This English version is the reference.

**How to open:** `conda activate hydroscope && jupyter lab labeling/label.ipynb`, run all cells,
type your annotator id and press Start. Every click is saved immediately; Start resumes where you
left off. Labels made with guide v1 are archived in `data/ags/labels_v1_2026-10-09.csv`.

## Context: what April looks like in Aguascalientes

- Semi-arid climate, about 510–530 mm of rain per year, falling mostly from June to October.
  **April is the peak of the dry season**, so most of the state looks beige or brown in the preview.
- **Green in April means water:** irrigated crops (alfalfa, forage maize, oats), orchards, the
  oak and pine forest of the western ranges, or vegetation along channels and dams.
- About 70 000 ha of rainfed (temporal) fields are **bare or fallow in April**; they are planted
  only after the rains start.
- Natural cover is mostly dry grassland and xeric scrub (crasicaule scrub, huizache, wild nopal)
  on the plains and hills, and oak/pine forest in the Sierra Fría and the western ranges.
- The **preview (April 2024) is the primary evidence**. The Esri basemap is recent imagery, often
  from another season: use it only to recognise objects (rows, fences, roofs, roads).

## The 10 classes (EuroSAT names, unchanged)

| Class | Label it when you see | Not this class |
| --- | --- | --- |
| AnnualCrop | A pattern of field parcels: straight boundaries, furrows, uniform tones per parcel. Includes **bare or fallow rainfed fields**, irrigated forage maize and oats, **alfalfa** and centre pivots. | Rows of trees or vines (PermanentCrop). |
| PermanentCrop | **Only orchards and vineyards in rows**: guava (mainly Calvillo), grapes, peaches, and nopal or maguey planted in rows. Regular dots of tree crowns. | Green uniform parcels without visible rows (AnnualCrop). |
| Pasture | Grass **managed for grazing**: fenced paddocks, uniform grass cover, regular shapes, often with a water trough or pond (bordo), no furrows. | Grass without signs of management (HerbaceousVegetation). |
| HerbaceousVegetation | Natural dry grassland, **xeric scrub** (crasicaule, huizache, wild nopal), open oak woodland where grass covers most of the ground, including rocky hills with sparse scrub. Irregular, patchy texture. | Parcels with straight boundaries (AnnualCrop or Pasture). |
| Forest | **Continuous tree canopy** of oak or pine over at least 70 % of the cell (Sierra Fría, western ranges). Dark green even in April. | Scattered trees over grass (HerbaceousVegetation). |
| Residential | Houses and dense small roofs with a street grid, including new subdivisions with streets already laid out. | Large roofs and yards (Industrial). |
| Industrial | Large warehouses and factory roofs, industrial parks, big parking and loading yards. | Houses (Residential). |
| Highway | A main road (federal highways, ring roads) crossing the cell and dominating the image. | Urban streets inside a neighbourhood (Residential). |
| River | A channel that dominates the cell, wet or dry, recognisable by its riparian vegetation or sandy bed. Rare. | A small ditch or canal inside fields (AnnualCrop). |
| SeaLake | **Water covering at least 70 % of the cell** in the April preview: dams (presas) and ponds (bordos). | A dam whose water has retreated: label the visible cover, or skip. |

## Rules

1. Label a cell only if **at least 70 %** of it is one class. Exception: Highway and River are
   labelled by the dominant element, as in EuroSAT.
2. Tie-breakers for the frequent semi-arid confusions:
   - Parcel pattern (straight boundaries, furrows), green or bare → **AnnualCrop**.
   - Trees or vines **in rows** → **PermanentCrop**; anything else is not PermanentCrop.
   - Grass with clear signs of grazing management → **Pasture**; without them →
     **HerbaceousVegetation**.
   - Tree canopy continuous over 70 % of the cell → **Forest**; otherwise
     **HerbaceousVegetation**.
3. **Hills and rocky ground:** relief is not a class; label the cover on it. Scrub or grass with
   rock or soil visible between plants → **HerbaceousVegetation**; continuous oak/pine canopy →
   **Forest**; **skip** only when bare rock or soil covers most of the cell (≥ 70 %), or for
   quarries and gravel pits (bancos de material).
4. Press **skip** for cover that has no EuroSAT class: greenhouses, quarries and mines, airports,
   golf courses, landfills, surfaces with almost no vegetation (bare rock outcrops, eroded
   gullies/cárcavas, bare soil without a parcel pattern), and any cell where you hesitate.
5. **back** returns to the previous cell; clicking a class again replaces your earlier label.

## Examples (fill in while labeling)

| Class | cell_id 1 | cell_id 2 |
| --- | --- | --- |
| AnnualCrop | | |
| Forest | | |
| HerbaceousVegetation | | |
| Highway | | |
| Industrial | | |
| Pasture | | |
| PermanentCrop | | |
| Residential | | |
| River | | |
| SeaLake | | |
