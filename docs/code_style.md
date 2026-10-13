# Code style: write it the way we saw it in class

This project is the final assignment of the Deep Learning module. Its code must be readable by a
student who is **still learning Python**: the same person who followed the course notebooks has to
understand, run and defend every file in `src/hydroscope/`. The notebooks are therefore the style
reference, not a Python style guide for professional libraries.

The reference notebooks live in the course folder (path in `CLAUDE.local.md`, never in git):

| Notebook | What we take from it |
| --- | --- |
| `CNNs_Fundamentals_PyTorch.ipynb` | The project's main model: plain PyTorch, `nn.Sequential` blocks, `run_epoch` / `fit` loop, `count` of parameters, one `print` line per epoch, matplotlib curves |
| `Introduction_to_PyTorch/Introduction_to_PyTorch_empty.ipynb` | `nn.Module` with `__init__` + `forward`, a `Dataset` with `__init__` / `__len__` / `__getitem__`, `train_model` / `eval_model`, `state_dict` save and load |
| `03_CNNs Backbones.zip` → `Inception_ResNet_DenseNet.ipynb` | `set_seed`, mean/std computed on the **train** set only, `train_transform` vs `test_transform`, a `model_dict` + `create_model(name, ...)` factory, `optim.AdamW`, LR scheduler |
| `Logistic_Regression_Iris_dataset.ipynb`, `Linear-Regression/*.ipynb` | `sklearn.metrics` (`accuracy_score`, `confusion_matrix`), pandas for tables |
| `RNN__LSTM.ipynb`, `LSTM_TS.ipynb` | `torch.no_grad()` for evaluation, `pathlib.Path`, `TensorDataset` / `DataLoader` |

The slides (`01_...pdf` to `04_...pdf`, `Activation_Functions.pdf`) define *what* we may use
(architectures, normalization, augmentation, metrics, Grad-CAM, self-supervised weights). The
notebooks define *how* the code should look.

## 1. The shape of a module

Every file in `src/hydroscope/` follows the notebook order: imports, constants, small functions,
then (if needed) one `main()`.

```python
"""One sentence saying what this file does."""

import torch
import torch.nn as nn

CLASSES = ["AnnualCrop", "Forest", ...]   # constants in UPPER_CASE, right after the imports


def count_parameters(model):
    # Total number of trainable parameters, as in the CNN notebook.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- **Functions first.** A function receives plain arguments and returns a value. Classes are used
  only where PyTorch asks for them: `nn.Module` and `torch.utils.data.Dataset`.
- **One idea per function, under ~40 lines.** If a function needs a second screen, split it.
- **Names say what the thing is**: `train_loader`, `val_macro_f1`, `best_state_dict`. No one-letter
  names except `x`, `y`, `i`, `n` in short loops, exactly like the notebooks.

## 2. Python we use (seen in class)

- `def`, `return`, `if / elif / else`, `for ... in`, `while`, `with torch.no_grad():`
- lists, dicts, tuples, f-strings, `enumerate`, `zip`, `range`, `len`, `sum`, `max`, `min`
- simple list comprehensions: `[c for c in CLASSES]`; never nested, never with `if/else` inside
- `pathlib.Path`, `json`, `csv` / `pandas`, `numpy`, `matplotlib`
- type hints with built-in types only: `int`, `float`, `str`, `bool`, `list[str]`,
  `dict[str, float]`, `torch.Tensor`, `Path`, `nn.Module`. Use `| None` for optional values.
- docstrings: one to three plain sentences. Comments explain **why** and write tensor shapes
  as in class: `# (N, 13, 64, 64) -> (N, 13, 224, 224)`
- errors: `raise ValueError("bands must be 'rgb' or 'ms13'")` with a message a student understands

## 3. Python we avoid (not seen in class)

| Avoid | Use instead |
| --- | --- |
| `from __future__ import annotations`, `typing.Any`, `Mapping`, `Sequence`, `Iterable`, `TypeVar`, `Protocol`, `Callable` | built-in types; `list`, `dict`; or no hint if it gets long |
| `@dataclass`, `@property`, `@staticmethod`, `@classmethod`, custom decorators | plain functions and plain attributes |
| `lambda` assigned to a name, dicts of lambdas | a small `def` |
| generators (`yield`), `itertools`, `functools`, `contextlib` | a loop that builds a list |
| dunder tricks (`__getstate__`, `__getattr__`), metaclasses, inheritance beyond `nn.Module` / `Dataset` | explicit code |
| `*args, **kwargs` pass-through | named arguments (the only exception is the call to `timm.create_model`) |
| `logging` with handlers and formatters | `print(f"epoch {epoch:2d} | train loss {loss:.3f} | val macro-F1 {f1:.3f} | {secs:.0f}s")` |
| `argparse` with many flags or subcommands | `--config path.yaml` plus at most two or three flags |
| atomic writes, temp files, `os.replace`, mmap tricks, streaming statistics | the direct version (`np.save`, `np.load`, `.mean()`), unless memory makes it impossible, and then a comment says why |
| `assert` in library code | `raise ValueError(...)`; `assert` belongs in tests |
| PyTorch Lightning, Hydra, Weights & Biases, `torch.compile`, autocast / mixed precision | the plain loop below, YAML read with `yaml.safe_load`, CSV logs, fp32 |

If a construct from the left column is already in the repo and works, leave it; only replace it
when you are editing that function anyway and the simple version behaves exactly the same.

## 4. Patterns to reuse verbatim

**Seed (Tutorial 5):**

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

**Dataset (Tutorial 2):**

```python
class AguascalientesDataset(Dataset):
    def __init__(self, labels_csv, patches_dir, bands, split):
        ...

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = ...          # float32 (C, 64, 64), raw digital numbers
        label = ...          # int
        return image, label
```

**Model factory (Tutorial 5, `model_dict` + `create_model`), written with timm:**

```python
TIMM_NAMES = {"resnet50": "resnet50", "efficientnet_b0": "efficientnet_b0",
              "vit_small": "vit_small_patch16_224"}


def build_model(name, bands, init="imagenet", num_classes=10):
    # Same function for the three models: only the architecture changes.
    if name not in TIMM_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {list(TIMM_NAMES)}")
    in_chans = 3 if bands == "rgb" else 13
    model = timm.create_model(TIMM_NAMES[name], pretrained=(init == "imagenet"),
                              num_classes=num_classes, in_chans=in_chans)
    ...
    return model
```

**Training loop (CNN notebook `run_epoch` / `fit`):**

```python
def run_epoch(model, loader, criterion, optimizer=None):
    # One pass over the data. Trains if an optimizer is given, otherwise only evaluates.
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            all_preds += logits.argmax(dim=1).tolist()
            all_labels += y.tolist()
    return total_loss / len(all_labels), all_preds, all_labels
```

Gradient accumulation (protocol: batch 64 = 2 micro-batches of 32) is the only addition to this
loop: divide `loss` by `accum_steps`, call `optimizer.step()` every `accum_steps` micro-batches.
Write it inside `run_epoch`, with a comment, not in a helper class.

**Metrics (Iris notebook):**

```python
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
f1_per_class = f1_score(all_labels, all_preds, average=None, labels=range(10))
confusion = confusion_matrix(all_labels, all_preds, labels=range(10))
```

**Checkpoints (Tutorial 2):** `torch.save(model.state_dict(), path)` and
`model.load_state_dict(torch.load(path, map_location="cpu"))`. The companion `.yaml` is the run
config written with `yaml.safe_dump`.

**Curves (CNN notebook):** a `history` dict of lists → `pandas.DataFrame(history).to_csv(...)` →
matplotlib `plot` with `"-o"` markers, grid at `alpha=0.3`, one figure saved as PDF.

## 5. Tests

- One behavior per test, `assert` on shapes, values or raised errors; tiny tensors (`torch.zeros(2, 13, 64, 64)`).
- A test reads like a sentence: `test_build_model_ms13_returns_10_logits`.
- No fixtures beyond `tmp_path`; no parametrize with more than a handful of cases; no mocks unless
  the real thing needs the network (Earth Engine, TorchGeo download).

## 6. Libraries outside the notebooks

These are in the plan and allowed, each for one job: **timm** (the three architectures),
**TorchGeo** (EuroSAT download and SSL4EO-S12 weights), **pytorch-grad-cam** (Grad-CAM),
**rasterio / geopandas / earthengine-api** (stage-1 data pipeline, already written). Anything else
is a proposal: mark it as optional in the PR and explain in one line what it replaces.
