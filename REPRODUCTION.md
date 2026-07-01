# Bits2Bites — Provenance & Cleanup Log

This repository is the original Bits2Bites research code **rebased onto a clean
Pointcept v1.7.0** and cleaned up for reproducibility. It replaces the earlier
fork, whose training/testing workflow had accumulated confused, one-off edits.

- **Base:** Pointcept `v1.7.0` (commit `d727225`).
- **Branch:** `bits2bites-clean`.

## Kept (genuine contributions, ported unchanged then tidied)

- `pointcept/models/multi_task_classifier/` — `MultiTaskClassifier` (backbone +
  5 heads + `stl_task` loss-zeroing).
- `pointcept/datasets/dental.py` — `DentalDataset`.
- `pointcept/datasets/preprocessing/dental/` — mesh→pointcloud + landmark prep.
- `configs/dental/` — refactored (see below).
- `tools/ToothFairy4M_class_weights.py`, `tools/dental_fold.py`.
- `pointcept/engines/hooks/evaluator.py::MultiClsEvaluator` — per-task metrics hook.

## Changed vs. the original fork (and why)

- **`cls_mode` → `enc_mode`.** Pointcept renamed the encoder-only flag; the
  SpUNet global-mean-pool that the fork added by hand is now upstream identical.
  Fixed in the configs; **no backbone edits needed**.
- **`index_valid_keys` added to `DentalDataset`.** v1.7.0 subsamples only keys
  listed in `index_valid_keys`; without it `point_label_onehot` would not be
  downsampled alongside `coord`. Now set to `["coord", "point_label_onehot"]`.
- **Removed the UUID data-path hack** in `tools/train.py` (it overrode every
  config's `data_root` with `data/dt_<uuid>`). `tools/train.py` is now stock
  Pointcept; fold assembly is an explicit step (`tools/dental_fold.py`).
- **Added `MultiClsTester`** (`pointcept/engines/test.py`), registered in
  `TESTERS`, sharing the metric helper with `MultiClsEvaluator`. It replaces the
  incompatible `ClsTester` path and retires the ad-hoc `tools/infer_dental.py`
  (writes `result/metrics.json` + `result/predictions.json`).
- **Config refactor.** The 12 near-duplicate `configs/dental/*.py` (differing by
  one line) now inherit from `configs/dental/_base_dental.py`; backbone configs
  set only the model/optimizer, STL configs set only `model.stl_task`. Values are
  unchanged. Standalone `test` now evaluates the labelled held-out (val) fold.
- **`tools/dental_fold.py`** decoupled from training, given a CLI, and now clears
  stale `dental_*.pth` caches when re-folding (a latent bug once `data_root` is
  fixed across runs).
- **Dead code / i18n.** Removed the commented TTA `_prepare_test_item`, unused
  `num_points`/`uniform_sampling`/FPS args, and the unused `pointops` import from
  `DentalDataset`; comments normalized to English (label vocabularies kept in the
  original Italian to match the annotation CSV).
- **`enable_wandb=False`** by default.
- **Dependency management moved to uv** (`pyproject.toml`); `environment.yml`
  (conda) removed. `pandas`/`scikit-learn` are now declared as base deps, with
  GPU/compiled ops (`torch-scatter`, `spconv-cu124`, `flash-attn`, `libs/pointops`,
  `libs/pointgroup_ops`) under the `[gpu]` extra.

## Reproduction caveat

Because the code was rebased onto a newer Pointcept, exact numeric parity with
the published tables is not guaranteed. Per-fold macro-F1 should land in the
paper's ballpark (PTv3 MTL ≈ 0.63); a large gap indicates a regression to chase.
See `BITS2BITES.md` for how to run.
