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
(no conda). Three steps: a base env (works anywhere), then GPU/compiled extras
(need a CUDA 12.4 toolkit + GPU).

```bash
# 1) base env (CPU-installable: torch cu124, numpy, pandas, scikit-learn, ...)
uv venv --python 3.10
uv sync                       # installs [project.dependencies]

# 2) GPU/compiled extras — on a CUDA-12.4 machine with nvcc + a GPU:
#    torch-scatter, spconv-cu124, and the in-repo libs/pointops
uv sync --extra gpu
uv pip install ./libs/pointops --no-build-isolation

# (optional) flash-attn, only if you set enable_flash=True in a PTv3 config:
#    uv pip install -e '.[flash]' --no-build-isolation
```

`uv sync` pins torch/torchvision/torchaudio from the cu124 index configured in
`pyproject.toml`. The `[gpu]` extras compile against that torch, so they must be
installed **after** the base env and **cannot** be built on a machine without
CUDA. If a wheel is unavailable for your platform, install that package's build
prerequisites first (see its upstream docs).

## Data layout

The config's `data_root` (default `data/dental_landmarks_mesh`) expects:

```
data/dental_landmarks_mesh/
    fold_1/  dental_<id>.json ...
    fold_2/  ...
    ...  fold_5/
    labels.csv          # columns: id, right, left, anterior, transverse, midline
```

Each `dental_<id>.json` has `objects`, each with a `coord` (xyz) and a `class`
(`"Mesh"` or one of Mesial/Distal/Cusp/FacialPoint/OuterPoint/InnerPoint).

### Building it from the raw download

The public dataset (`datasets/Bits2Bites/`) ships 200 patients as
`<id>/{lower,upper}.stl` plus a top-level `Annotations.csv` — **meshes only,
no per-tooth landmarks**. Turn this into the layout above with:

```bash
python pointcept/datasets/preprocessing/dental/prepare_bits2bites.py \
    --dataset-root <Downloaded Bits2Bites dataset> \
    --output-dir data/dental_landmarks_mesh
```

This translates `Annotations.csv` into the Italian label vocabulary
`pointcept/datasets/dental.py` expects, builds a `"Mesh"`-only point cloud per
patient from the STL scans, and stratifies everything into `fold_1..fold_5` +
`labels.csv` via `create_balanced_folds` (the same routine
`pointcept/datasets/preprocessing/dental/preprocess_dentalnet.py` uses for
other label sources). This mesh-only variant has no per-point landmark
one-hot bits set (all zero) — every point is class `"Mesh"`.

**Reproducing the paper's full model (mesh + landmarks).** The paper's actual
model uses 9-D points (`coord` + a 6-D per-tooth landmark one-hot from a
3DTeethLand-style landmark/segmentation predictor — see "Input" above); that
predictor is external and not included in this repo. If you have landmark
annotations for the patients, drop them in at
`datasets/Bits2Bites/landmarks/dental_<id>.json` (one file per patient, IDs
zero-padded to 4 digits, e.g. `dental_0001.json`) with this schema:

```json
{
  "version": "1.1",
  "description": "landmarks",
  "key": "dental_0001",
  "objects": [
    {"key": "<any unique string>", "class": "Mesial", "coord": [x, y, z]},
    {"key": "<any unique string>", "class": "Distal", "coord": [x, y, z]}
  ]
}
```

`class` must be one of `Mesial`, `Distal`, `Cusp`, `FacialPoint`,
`OuterPoint`, `InnerPoint` (the 6 landmark types in `pointcept/datasets/
dental.py`'s `POINT_CLASSES`), combining both jaws' landmarks into one file
per patient. When `prepare_bits2bites.py` finds this folder it automatically
merges the landmarks with the mesh points instead of building a mesh-only
sample — no flags needed.

### Mesh-only / landmark-only variants

Once you have a complete `landmarks+mesh` data_root (e.g.
`data/dental_landmarks_mesh`, from the step above), derive either single-
modality variant from it — same `labels.csv`/fold split, same configs, just
fewer points per sample. Useful for a quick smoke test: landmark-only is
~240 points/patient vs ~198k for mesh-only, so it trains far faster.

```bash
# landmarks only (drop "Mesh" points)
python pointcept/datasets/preprocessing/dental/filter_landmarks_only.py \
    --input-dir data/dental_landmarks_mesh --output-dir data/dental_landmarks_only

# mesh only (drop the 6 landmark classes)
python pointcept/datasets/preprocessing/dental/filter_mesh_only.py \
    --input-dir data/dental_landmarks_mesh --output-dir data/dental_mesh_only
```

Then point `tools/dental_fold.py --data-root` at the new folder, and pass
the same path via `scripts/train.sh -e <data_root>` (overrides
`data.train/val/test.data_root` in the shared config through `--options`;
`test.sh` needs no flag — it reloads the saved `config.py` snapshot, which
already has the override baked in). See
`scripts/sbatch_train_fold1_landmarks_only.sh` /
`scripts/sbatch_train_fold1_mesh_only.sh` for ready-to-submit SLURM jobs.
Mesh-only samples get an all-zero landmark one-hot; landmark-only samples
have every point tagged as one of the 6 landmark classes.

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

## Weights & Biases (wandb) setup

`enable_wandb` is `True` by default (`configs/_base_/default_runtime.py`). Three
env vars control where runs are logged:

| Var | Purpose | Read by |
|---|---|---|
| `WANDB_API_KEY` | auth token (wandb.ai account settings) | wandb SDK directly |
| `WANDB_ENTITY` | team/user to log under, e.g. `maxillo` | wandb SDK directly |
| `WANDB_PROJECT` | project name, e.g. `b2bv2` | `configs/dental/_base_dental.py` (`wandb_project`, falls back to `"bits2bites"` if unset) |

Without `WANDB_ENTITY`, `wandb.init()` fails with `entity not specified, and
viewer has no default entity` if the account has no default entity set.

Export all three from a **shared, non-git-tracked env file** rather than
hardcoding a key in a script — on this cluster, `/work/grana_maxillo/lborghi/env.sh`
(already `source`d by every `scripts/sbatch_*.sh`):

```bash
export WANDB_API_KEY=<your-key>
export WANDB_ENTITY=<your-entity>
export WANDB_PROJECT=<your-project>
```

Alternative: run `wandb login` once on the login node (uses `$HOME/.netrc`) —
still requires `WANDB_ENTITY` set if the account belongs to multiple teams or
has no default.

To disable wandb entirely, set `enable_wandb = False` in a config.

## Training settings (from the paper)

| | PTv3 | SpUNet |
|---|---|---|
| optimizer | AdamW, lr 1e-4, wd 0.01 | SGD, lr 1e-2, wd 1e-4, nesterov |
| scheduler | CosineAnnealingLR | MultiStepLR [0.6, 0.8] γ0.1 |
| embed dim | 128 | 256 |

Shared: 200 epochs, batch size 8, grid size 0.01, mixed precision, grad-clip 1.0,
augmentations (normalize, scale, shift, z-rotate, dropout). Primary metric:
task-averaged macro-F1. Reference: PTv3 MTL ≈ 0.63 macro-F1 (per-fold results vary).
