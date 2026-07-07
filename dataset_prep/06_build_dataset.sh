#!/bin/bash
# Step 4 — turn the oriented + landmarked raw dataset into a DentalDataset
# data_root the existing BITS2BITES.md training commands can consume unchanged.
#
# This reuses the repo's own prepare_bits2bites.py (run from the MAIN Bits2Bites
# .venv, since it imports pointcept preprocessing), NOT the dataset_prep venv.
#
# Usage:
#   ORIENTED=<oriented dataset> OUTDIR=data/dental_landmarks_mesh_v2 bash 06_build_dataset.sh
set -euo pipefail

REPO_DIR=/work/grana_maxillo/lborghi/code/Bits2Bites
cd "$REPO_DIR"
source "$REPO_DIR/.venv/bin/activate"

ORIENTED=${ORIENTED:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented}
OUTDIR=${OUTDIR:-data/dental_landmarks_mesh_v2}

python pointcept/datasets/preprocessing/dental/prepare_bits2bites.py \
    --dataset-root "$ORIENTED" \
    --output-dir "$OUTDIR"

# Optional single-modality variants (same labels.csv / fold split):
if [ "${MAKE_VARIANTS:-0}" = "1" ]; then
    python pointcept/datasets/preprocessing/dental/filter_landmarks_only.py \
        --input-dir "$OUTDIR" --output-dir "${OUTDIR}_landmarks_only"
    python pointcept/datasets/preprocessing/dental/filter_mesh_only.py \
        --input-dir "$OUTDIR" --output-dir "${OUTDIR}_mesh_only"
fi

echo "Built data_root -> $OUTDIR"
echo "Train with e.g.:"
echo "  python tools/dental_fold.py --fold-val 1 --data-root $OUTDIR"
echo "  sh scripts/train.sh -d dental -c cls-ptv3-base -n ptv3_mtl_fold1_v2 -g 1 -e $OUTDIR"
