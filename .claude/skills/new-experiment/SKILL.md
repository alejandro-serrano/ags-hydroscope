---
name: new-experiment
description: Create a run config from the shared protocol for one model, band setting and initialization, then print the exact command to launch it. Usage - /new-experiment <resnet50|efficientnet_b0|vit_small> <rgb|ms13> <imagenet|ssl4eo>
disable-model-invocation: true
---

Arguments: $ARGUMENTS (model, bands, init).

1. Validate the arguments. `ssl4eo` is only valid with `ms13` and with `resnet50` (MoCo) or
   `vit_small` (DINO); refuse other combinations and explain why.
2. Create `configs/<model>_<bands>_<init>.yaml` that `extends: base.yaml` and sets ONLY
   `model`, `bands`, `init` and `run_id`. Never override a shared hyperparameter.
3. Run the config in smoke mode on CPU (`--smoke`, 200 images, 1 epoch) and confirm a result JSON is
   written to a temporary folder (not to `experiments/results/`).
4. Print the full-run command for the user, plus the estimated time from `docs/compute.md` if it
   exists. Do not launch the full run.
