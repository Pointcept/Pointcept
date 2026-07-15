# Dataset-expansion pipeline

Turns a new clinical export (`Bits2Bites2` layout) into an **expansion of** the
existing Bits2Bites dataset and produces a `data_root` the `BITS2BITES.md`
training commands consume unchanged. The export is *merged into* Bits2Bites
(not converted standalone), oriented into one canonical frame, and given
per-tooth landmarks by an external predictor.

```
Bits2Bites (A) ─┐
                 ├─► 01 merge ─► 02 dedup ─► 03 orient ─► 04/05 landmarks ─► 06 build ─► train
Bits2Bites2 (B) ─┘   (CPU)        (CPU)        (GPU)          (GPU + CPU)      (CPU)
```

The login node has **no GPU**. Steps 01, 02, 04, 05, 06 run on it. Steps 03
(orientation) and the 3DTeethLand inference in 04/05 go to a GPU node via the
`sbatch_*.sh` wrappers (partition `all_usr_prod`, account `grana_maxillo`, one
GPU, no specific GPU model required).

## 0. One-time environment

A single shared `dataset_prep/.venv` (created with `uv`), **separate from the
main `Bits2Bites/.venv`** so the training env is left untouched:

Important: you need a CUDA 12.6+ capable machine. On a cluster, make sure to "module load cuda/12.6"

```bash
bash dataset_prep/setup_env.sh
```

Inside note: see build_dataset_prep_venv.sbatch

Installs torch (cu124) + `trimesh` + `numpy` + `matplotlib` for the orientation
net (`ios_orienter`, vendored at `dataset_prep/ios_orienter/`, needs only
`numpy` + `torch`), **and** BracketPrediction's inference-only deps (`spconv-cu124`,
`torch-scatter`/`torch-cluster`/`torch-geometric`, `pointops` built from
BracketPrediction's `libs/pointops`, plus einops/timm/etc.) — see `setup_env.sh`
for the full list and the reasoning behind what's included/excluded. This is now
a single shared venv for both dataset-prep GPU steps; there is no separate
BracketPrediction venv.

**Model weights** are vendored locally under `weights/` (git-ignored):
`orient_pointnet_best.pt` (ios_orienter orientation), `segmentator_best.pth`
(3DTeethLand segmentation), `heatmap_landmarks.pth` (3DTeethLand bond/landmark
prediction). The `sbatch_*.sh` scripts default to these, so the pipeline needs
no access to anyone else's home directory. `orient_pointnet_best.pt` ships with
the `ios_orienter` repo itself (`ios_orienter/weights/pointnet_best.pt`, trained
via `ios_orienter/train.py`) — no external source path. `segmentator_best.pth`
originally from `Mlugli/models/AutoBonding/segmentator_best.pth`.
`heatmap_landmarks.pth` from
`/work/grana_maxillo/averonese_STS2026/BracketPrediction/checkpoints/heatmap_landmarks.pth`
(copied in manually, not scripted); this supersedes the old `bond_model_best.pth`
(originally from `Mlugli/models/Brackets/3dteethland_0/model/model_best.pth`),
which is now stale.

## Run it all

```bash
bash dataset_prep/run_all.sh
```

It runs the CPU steps, then stops and prints the exact `sbatch` line whenever a
GPU step is next. Submit that job, wait for it, and re-run `run_all.sh` — it
detects the finished output and continues. Paths are overridable via env vars
(`B1 B2 MERGED ORIENTED LM_IN LM_OUT OUTDIR`); defaults are under
`/work/grana_maxillo/lborghi/datasets/`.

Below is the same thing step by step.

## 1. Merge + convert  (`01_merge_convert.py`, CPU)

```bash
python dataset_prep/01_merge_convert.py \
    --bits2bites  /work/grana_maxillo/lborghi/datasets/Bits2Bites \
    --bits2bites2 /work/grana_maxillo/lborghi/datasets/Bits2Bites2 \
    --output      /work/grana_maxillo/lborghi/datasets/Bits2Bites_merged --force
```

- Copies A (patients 1..200) verbatim, appends B renumbered from 201.
- Translates B's `classification.json` `manual` block into A's `Annotations.csv`
  vocabulary. B scans whose label is **not** `manual` (auto-generated, not
  expert-reviewed) are dropped and listed by their **original folder name** (which
  may embed a real patient name) in `reports/dropped_no_manual.csv` so the doctor
  can re-annotate.
- **Label space is identical to Bits2Bites** (heads: right 3 / left 3 / anterior 4
  / transverse 3 / midline 2). B's `II_edge` and `II_full` both collapse to
  *seconda classe*, exactly as A does. Any axis marked `Unknown` (unevaluable —
  present in B on vertical/transverse/midline, and in A on sagittal) becomes an
  ignored label (`-1`) for that head only; the sample is still used for its other
  heads. This required a one-line `Unknown` fallback in the repo's
  `prepare_bits2bites.py` (`ANTERIOR_MAP` / `MIDLINE_MAP` / `map_transverse`).
- Intra-oral photos are copied to `photos/<id>/` (all formats incl `.heic`) — kept
  aside for a future extension; the training path ignores them.
- `reports/id_mapping.csv` maps each new id → source + original folder
  (**sensitive**: contains real names).

## 1.2 De-duplicate  (`02_dedup.py`, CPU)

```bash
python dataset_prep/02_dedup.py --dataset /work/grana_maxillo/lborghi/datasets/Bits2Bites_merged
```

SHA-256 of each `(lower.stl, upper.stl)` pair; identical pairs collapse to one,
keeping the lowest id (so a published A patient always wins over a B copy). Kept
patients are renumbered contiguously (A's 1..200 unchanged); folders, photos, and
landmark files are renumbered together. Removals → `reports/dedup_removed.csv`.
Add `--output <dir>` for a non-destructive copy instead of editing in place.

## 2. Orientation  (`03_orient.py` via `sbatch_orient.sh`, GPU)

```bash
MERGED=/work/grana_maxillo/lborghi/datasets/Bits2Bites_merged \
ORIENTED=/work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented \
sbatch dataset_prep/sbatch_orient.sh
```

Orients the **whole** merged set into one canonical frame with the `ios_orienter`
PointNet regressor, using the local checkpoint `weights/orient_pointnet_best.pt`.

**Upper/lower stay locked together:** the rotation is predicted **once from the
merged upper+lower point cloud** (ios_orienter's most accurate mode) and applied
to both meshes. **Only the rotation is applied — original mm dimensions and
position are preserved** (`oriented = (v-center) @ matrix + center`, `matrix` a
proper rotation), so the scans keep their real-world size, not a unit-sphere
rescale. A's existing landmarks are carried through the *same* rotation;
per-patient transforms are saved to `transforms/<id>.json` (invertible:
`v = (oriented-center) @ matrix.T + center`), including a `confidence_deg`
field (ios_orienter's test-time-augmentation agreement — lower is more
confident; a good field to sort/filter on to spot likely-bad orientations).

**QA — eyeball the imperfect net:** `qa/overview_before_after.png` shows 10
patients, two columns (before/after), upper (red) + lower (blue) overlaid. Open it
before trusting the result. `--qa-per-scan` also writes one PNG per patient.
`--limit N` processes only the first N (quick check).

## 3. Landmarks  (external 3DTeethLand predictor)

Landmarks are **not** manually annotated — they are inferred by BracketPrediction
(vendored in-repo at `dataset_prep/BracketPrediction/`, branch `new_landmarks`),
on the **oriented** meshes so they come back already in the training frame.

**3a. Stage input** (`04_landmarks_prep.py`, CPU) — only for patients lacking
landmarks (the B patients):

```bash
python dataset_prep/04_landmarks_prep.py \
    --dataset /work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented \
    --lm-in   /work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_in --force
```

**3b. Run inference** (`sbatch_3dteethland.sh`, GPU) — uses the vendored
BracketPrediction repo + the shared `dataset_prep/.venv`:

```bash
LM_IN=/work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_in \
LM_OUT=/work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_out \
sbatch dataset_prep/sbatch_3dteethland.sh
```

Uses BracketPrediction vendored at `dataset_prep/BracketPrediction/` (default
`BRACKET_DIR=$PREP_DIR/BracketPrediction`, override via env for an alternate
checkout) with the shared `dataset_prep/.venv` from `setup_env.sh` — no separate
BracketPrediction venv needed anymore. The seg + bond weights default to the
**local** copies in `weights/` (`segmentator_best.pth`, `heatmap_landmarks.pth`),
so no access to anyone else's home is needed at run time; the seg/bond
*configs* are in the BracketPrediction repo. The script also passes
`--preprocessing 3dteethland_preprocessing.yaml`, `--cache`, and `--save-ply`
for the creator-recommended inference path; override the preprocessing file with
`PREPROCESSING=/path/to/file.yaml` if needed.

**3c. Ingest** (`05_landmarks_ingest.py`, CPU) — parse `predictions.csv` into
`landmarks/dental_<id:04d>.json`, keeping only the six DentalDataset classes
(Mesial, Distal, Cusp, FacialPoint, OuterPoint, InnerPoint):

```bash
python dataset_prep/05_landmarks_ingest.py \
    --output-folder /work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_out \
    --landmarks-dir /work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented/landmarks
```

## 4. Build the data_root  (`06_build_dataset.sh`, CPU, main venv)

```bash
ORIENTED=/work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented \
OUTDIR=data/dental_landmarks_mesh_v2 \
bash dataset_prep/06_build_dataset.sh
```

Runs the repo's own `prepare_bits2bites.py` (from the **main** `Bits2Bites/.venv`)
to produce `data/dental_landmarks_mesh_v2/{fold_1..5, labels.csv}`. Set
`MAKE_VARIANTS=1` to also emit `_landmarks_only` / `_mesh_only`.

## 5. Train — the existing BITS2BITES.md commands, bigger dataset

```bash
cd /work/grana_maxillo/lborghi/code/Bits2Bites
python tools/dental_fold.py --fold-val 1 --data-root data/dental_landmarks_mesh_v2
sh scripts/train.sh -d dental -c cls-ptv3-base -n ptv3_mtl_fold1_v2 -g 1 -e data/dental_landmarks_mesh_v2
sh scripts/test.sh  -d dental -n ptv3_mtl_fold1_v2 -g 1
```

Everything downstream (configs, folds, model) is unchanged — only the data_root
is larger.

## Reports & artifacts

| File | Meaning | Sensitivity |
|------|---------|-------------|
| `<merged>/reports/dropped_no_manual.csv` | B scans dropped for no expert label, by original name | real names |
| `<merged>/reports/dedup_removed.csv` | duplicate scans removed | — |
| `<merged>/reports/id_mapping.csv` | new id → source + original folder | real names |
| `<oriented>/transforms/<id>.json` | per-patient orientation transform | — |
| `<oriented>/qa/overview_before_after.png` | orientation QA montage | — |
