---
name: bootstrap-repo
description: Scaffold the ags-hydroscope repository for task F0-01 - folders, environment.yml, pyproject with ruff and pytest, Makefile, .gitignore, README skeleton, docs/protocol.md and docs/contracts.md templates, smoke config and CI workflow.
disable-model-invocation: true
---

Create the stage-1 skeleton of the repository, following CLAUDE.md. Do not overwrite existing files;
if a file exists, show what you would change instead.

1. Folders: `src/hydroscope/{data,models,training,explain,geo}`, `configs/`, `docs/`, `data/mock/`,
   `experiments/results/`, `notebooks/`, `scripts/`, `tests/`, `paper/{tables,figures}`, `slides/`,
   `labeling/`, each package with `__init__.py`.
2. `environment.yml` (name `hydroscope`, Python 3.11): pytorch, torchvision, timm, torchgeo,
   torchmetrics, grad-cam, rasterio, geopandas, earthengine-api, pandas, pyarrow, matplotlib,
   ipywidgets, pyyaml, pytest, ruff. Pin major versions only.
3. `pyproject.toml` with the package, ruff (line length 100) and pytest settings.
4. `Makefile` targets: `env`, `lint`, `test`, `smoke`, `train CONFIG=`, `eval CKPT= SPLIT=`,
   `tables`, `figures`.
5. `.gitignore`: data (except `data/mock/`), checkpoints, rasters, `.env`, `CLAUDE.local.md`,
   notebooks checkpoints.
6. `docs/protocol.md` and `docs/contracts.md` filled from the invariants in CLAUDE.md.
7. `configs/base.yaml` (shared protocol), `configs/smoke.yaml` (CPU, 200 images, 1 epoch).
8. `.github/workflows/ci.yml`: ruff + pytest on push and pull_request.
9. `README.md` skeleton in English: overview, setup, reproduce, results, citation, license (MIT).

Finish by running `ruff check .` and `python -m pytest -q` and report the result.
