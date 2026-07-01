# Bits2Bites — Intra-oral Scans Occlusal Classification

Multi-task classification of dental occlusion from paired intra-oral scans (IOS),
built on [Pointcept](https://github.com/Pointcept/Pointcept). A single point-cloud
backbone with five classification heads jointly predicts five clinical occlusal
traits. This document is the reproduction guide; see `REPRODUCTION.md` for the
provenance of this cleaned-up codebase.

Paper: *Bits2Bites: Intra-oral Scans Occlusal Classification* (MICCAI 2025, ODIN
workshop). See `Bits2Bites_Intra_oral_Scans_.pdf`.

## Task

One combined upper+lower point cloud → five independent labels (`num_classes_list`):

| Head | Task              | Classes |
|------|-------------------|---------|
| 0    | right sagittal    | 3       |
| 1    | left sagittal     | 3       |
| 2    | anterior bite     | 4       |
| 3    | transverse bite   | 3       |
| 4    | midline           | 2       |

**MTL** trains one backbone + five heads jointly. **STL** trains one model per
task (loss on the other four heads is zeroed via `model.stl_task`).

## Model

`MultiTaskClassifier` (`pointcept/models/multi_task_classifier/`):
backbone (encoder-only) → global mean-pool over `offset` → five 2-layer MLP heads,
each a weighted `CrossEntropyLoss` (`ignore_index=-1`). Total loss = unweighted
mean of the five task losses. Backbones: `PT-v3m1` (PointTransformerV3) and
`SpUNet-v1m1`, both run with `enc_mode=True`.

## Input

Each point carries `[coord (3), point_label_onehot (6)]` = **9-D** features. The
6-D one-hot marks per-tooth landmark type (from the 3DTeethLand predictor); mesh
points get an all-zero landmark vector. For a landmark-only variant, prepare a
landmark-only dataset (mesh points dropped) and keep the same configs.

## Setup (uv)

Dependencies are managed with [uv](https://docs.astral.sh/uv/) via `pyproject.toml`
(no conda). Two steps: a base env (works anywhere), then GPU/compiled extras
(need a CUDA 12.4 toolkit + GPU).

```bash
# 1) base env (CPU-installable: torch cu124, numpy, pandas, scikit-learn, ...)
uv venv --python 3.10
uv sync                       # installs [project.dependencies]

# 2) GPU/compiled extras — on a CUDA-12.4 machine with nvcc + a GPU:
#    torch-scatter, spconv-cu124, and the in-repo libs/pointops etc.
uv pip install -e '.[gpu]' --no-build-isolation
# (optional) flash-attn, only if you set enable_flash=True in a PTv3 config:
#    uv pip install -e '.[flash]' --no-build-isolation
```

`uv sync` pins torch/torchvision/torchaudio from the cu124 index configured in
`pyproject.toml`. The `[gpu]` extras compile against that torch, so they must be
installed **after** the base env and **cannot** be built on a machine without
CUDA. If a wheel is unavailable for your platform, install that package's build
prerequisites first (see its upstream docs).

## Data layout

Place the dataset at the config's `data_root` (default `data/dental_landmarks_mesh`):

```
data/dental_landmarks_mesh/
    fold_1/  dental_<id>.json ...
    fold_2/  ...
    ...  fold_5/
    labels.csv          # columns: id, right, left, anterior, transverse, midline
```

Each `dental_<id>.json` has `objects`, each with a `coord` (xyz) and a `class`
(`"Mesh"` or one of Mesial/Distal/Cusp/FacialPoint/OuterPoint/InnerPoint).
Mesh→point-cloud + landmark preprocessing lives in
`pointcept/datasets/preprocessing/dental/preprocess_dentalnet.py`.

Class weights are already baked into `configs/dental/_base_dental.py`. To
recompute for new data, adjust the CSV path in
`tools/ToothFairy4M_class_weights.py` and run it.

## Running

Cross-validation folds are assembled explicitly (no hidden path rewriting). The
held-out fold is used for both validation-during-training and final testing (no
separate val split, matching the paper).

**Single fold, MTL, PTv3:**
```bash
python tools/dental_fold.py --fold-val 1 --data-root data/dental_landmarks_mesh
sh scripts/train.sh -d dental -c cls-ptv3-base -n ptv3_mtl_fold1 -g 1
sh scripts/test.sh  -d dental -n ptv3_mtl_fold1 -g 1
# metrics -> exp/dental/ptv3_mtl_fold1/result/metrics.json
#            predictions -> .../result/predictions.json
```

**Single-task (e.g. anterior bite = head 2):**
```bash
sh scripts/train.sh -d dental -c cls-ptv3-base-stl2 -n ptv3_stl2_fold1 -g 1
```

**Full 5-fold CV (train + test each fold):**
```bash
sh scripts/run_dental_cv.sh -c cls-ptv3-base -n ptv3_mtl -g 1 \
   -r data/dental_landmarks_mesh
```
Swap `-c cls-spunet-base` for the SpUNet backbone, or `cls-*-base-stlN` for STL.

## Training settings (from the paper)

| | PTv3 | SpUNet |
|---|---|---|
| optimizer | AdamW, lr 1e-4, wd 0.01 | SGD, lr 1e-2, wd 1e-4, nesterov |
| scheduler | CosineAnnealingLR | MultiStepLR [0.6, 0.8] γ0.1 |
| embed dim | 128 | 256 |

Shared: 200 epochs, batch size 8, grid size 0.01, mixed precision, grad-clip 1.0,
augmentations (normalize, scale, shift, z-rotate, dropout). Primary metric:
task-averaged macro-F1. Reference: PTv3 MTL ≈ 0.63 macro-F1 (per-fold results vary).

`enable_wandb=False` by default; set it `True` (and `wandb login`) to log runs.
