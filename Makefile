.PHONY: env lint test smoke train eval results tables figures paper release download

PYTHON ?= python
SPLIT ?= ags_test

env:
	conda env create -f environment.yml || conda env update -f environment.yml --prune

lint:
	ruff check . --fix && ruff format .

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m hydroscope.training.train --config configs/smoke.yaml

train:
	@test -n "$(CONFIG)" || (echo "usage: make train CONFIG=configs/<run>.yaml" && exit 1)
	$(PYTHON) -m hydroscope.training.train --config $(CONFIG)

eval:
	@test -n "$(CKPT)" || (echo "usage: make eval CKPT=checkpoints/<run>.pt [SPLIT=ags_test]" && exit 1)
	$(PYTHON) -m hydroscope.training.evaluate --ckpt $(CKPT) --split $(SPLIT)

# make_results.py writes the tables, the figures and the README results in one run.
results:
	$(PYTHON) scripts/make_results.py

tables figures: results

paper:
	tectonic paper/main.tex

# Build the release assets in dist/ (nothing is uploaded).
release:
	$(PYTHON) scripts/package_release.py

# Download the release assets to checkpoints/ and data/ags/, checking their SHA256.
download:
	$(PYTHON) scripts/download_release.py
