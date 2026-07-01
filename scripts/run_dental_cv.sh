#!/bin/bash
# 5-fold cross-validation driver for Bits2Bites occlusal classification.
#
# For each fold it (1) assembles train/val splits, (2) trains, (3) tests the
# held-out fold with MultiClsTester. Per-fold metrics land in
# exp/dental/<exp>_fold<k>/result/metrics.json.
#
# Usage:
#   sh scripts/run_dental_cv.sh -c cls-ptv3-base -n ptv3_mtl -g 1 \
#      -r data/dental_landmarks_mesh
#
# NOTE: folds share the same data_root and are run sequentially. To run folds in
# parallel, give each its own copy of data_root (and matching config data_root).

cd "$(dirname "$(dirname "$0")")" || exit
PYTHON=python
CONFIG="cls-ptv3-base"
EXP_NAME="dental_cv"
NUM_GPU=1
DATA_ROOT="data/dental_landmarks_mesh"

while getopts "p:c:n:g:r:" opt; do
  case $opt in
    p) PYTHON=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    n) EXP_NAME=$OPTARG ;;
    g) NUM_GPU=$OPTARG ;;
    r) DATA_ROOT=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG"; exit 1 ;;
  esac
done

echo "Config: $CONFIG | Exp: $EXP_NAME | GPUs: $NUM_GPU | data_root: $DATA_ROOT"

for fold in 1 2 3 4 5; do
  echo "========================= FOLD ${fold}/5 ========================="
  $PYTHON tools/dental_fold.py --fold-val "$fold" --data-root "$DATA_ROOT" || exit 1
  sh scripts/train.sh -p "$PYTHON" -d dental -c "$CONFIG" \
     -n "${EXP_NAME}_fold${fold}" -g "$NUM_GPU" || exit 1
  sh scripts/test.sh -p "$PYTHON" -d dental \
     -n "${EXP_NAME}_fold${fold}" -g "$NUM_GPU" || exit 1
done

echo "Done. Per-fold metrics: exp/dental/${EXP_NAME}_fold*/result/metrics.json"
