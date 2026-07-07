#!/bin/bash
# End-to-end driver for the Bits2Bites dataset-expansion pipeline.
#
# CPU steps (01 merge, 02 dedup) run here directly. The two GPU steps
# (orientation, landmark inference) must go to a GPU node via sbatch — this
# script runs the CPU parts, then STOPS and prints the sbatch commands, because
# the login node has no GPU. Re-run it after each GPU step completes; it detects
# finished outputs and continues.
#
# Override any path via env vars (see defaults below).
set -euo pipefail

PREP_DIR=/work/grana_maxillo/lborghi/code/Bits2Bites/dataset_prep
cd "$PREP_DIR"

B1=${B1:-/work/grana_maxillo/lborghi/datasets/Bits2Bites}
B2=${B2:-/work/grana_maxillo/lborghi/datasets/Bits2Bites2}
MERGED=${MERGED:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_merged}
ORIENTED=${ORIENTED:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented}
LM_IN=${LM_IN:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_in}
LM_OUT=${LM_OUT:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_out}
OUTDIR=${OUTDIR:-data/dental_landmarks_mesh_v2}

source "$PREP_DIR/.venv/bin/activate"

echo "== Step 1: merge + convert (CPU) =="
python 01_merge_convert.py --bits2bites "$B1" --bits2bites2 "$B2" --output "$MERGED" --force

echo "== Step 1.2: de-duplicate (CPU) =="
python 02_dedup.py --dataset "$MERGED"

if [ ! -f "$ORIENTED/Annotations.csv" ]; then
    echo
    echo ">> Step 2 (orientation) needs a GPU. Submit it, then re-run this script:"
    echo "     MERGED=$MERGED ORIENTED=$ORIENTED sbatch $PREP_DIR/sbatch_orient.sh"
    exit 0
fi
echo "== Step 2: orientation output present at $ORIENTED =="

echo "== Step 3a: stage landmark input (CPU) =="
python 04_landmarks_prep.py --dataset "$ORIENTED" --lm-in "$LM_IN" --force

if ! find "$LM_OUT" -name predictions.csv 2>/dev/null | grep -q .; then
    echo
    echo ">> Step 3b (landmark inference) needs a GPU. Submit it, then re-run this script:"
    echo "     LM_IN=$LM_IN LM_OUT=$LM_OUT sbatch $PREP_DIR/sbatch_3dteethland.sh"
    exit 0
fi
echo "== Step 3b: landmark predictions present under $LM_OUT =="

echo "== Step 3c: ingest landmarks (CPU) =="
python 05_landmarks_ingest.py --output-folder "$LM_OUT" --landmarks-dir "$ORIENTED/landmarks"

echo "== Step 4: build DentalDataset data_root (main venv) =="
ORIENTED="$ORIENTED" OUTDIR="$OUTDIR" bash 06_build_dataset.sh

echo
echo "Done. Train with:"
echo "  cd /work/grana_maxillo/lborghi/code/Bits2Bites"
echo "  python tools/dental_fold.py --fold-val 1 --data-root $OUTDIR"
echo "  sh scripts/train.sh -d dental -c cls-ptv3-base -n ptv3_mtl_fold1_v2 -g 1 -e $OUTDIR"
