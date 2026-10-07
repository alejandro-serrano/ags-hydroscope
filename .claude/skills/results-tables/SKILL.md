---
name: results-tables
description: Regenerate every paper table and figure from experiments/results/*.json - EuroSAT performance, Aguascalientes zero-shot vs fine-tuned, degradation and recovery, computational cost, convergence curves. Use whenever results change and before writing or revising the Results section.
---

1. Run (or create if missing) `scripts/make_tables.py` and `scripts/make_figures.py`. They read only
   `experiments/results/*.json` and write `paper/tables/*.tex` and `paper/figures/*.pdf`.
2. Tables (booktabs, 3 decimals for F1/accuracy, units in headers):
   - `eurosat.tex`: model × bands → accuracy, macro-F1.
   - `ags.tex`: model × bands × init → macro-F1 zero-shot, head, last_block.
   - `generalization.tex`: drop = F1_EuroSAT − F1_zero-shot; recovery = (F1_ft − F1_zero-shot) / drop.
   - `cost.tex`: params (M), train time (min), time/epoch (s), peak memory (GB), patches/s.
3. Figures: performance vs. cost scatter, convergence curves (val macro-F1 per epoch), per-class F1
   on Aguascalientes.
4. Report which files changed and any JSON that was skipped and why. Never type numbers by hand.
