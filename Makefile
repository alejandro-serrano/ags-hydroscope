.PHONY: env lint test smoke train eval tables figures

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

tables:
	$(PYTHON) scripts/make_tables.py

figures:
	$(PYTHON) scripts/make_figures.py
