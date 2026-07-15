#!/bin/bash
#SBATCH --partition=boost_usr_prod
# #SBATCH --constraint="gpu_L40S_45G"
#SBATCH --account=grana_maxillo
#SBATCH --gres=gpu:1
#SBATCH --job-name=b2b_landmarks
#SBATCH --output=b2b_landmarks_%j.out
#SBATCH --error=b2b_landmarks_%j.err
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

# Any GPU in all_usr_prod is fine; no --constraint on purpose.
# Runs the BracketPrediction 3DTeethLand landmark predictor on the staged input
# built by 04_landmarks_prep.py. BracketPrediction is now vendored in this repo
# at dataset_prep/BracketPrediction/ (own nested .git, branch new_landmarks —
# same vendoring pattern as dataset_prep/ios_orienter/). It uses the SHARED
# dataset_prep/.venv (see setup_env.sh), not a separate env.

set -euo pipefail

PREP_DIR=/work/grana_maxillo/lborghi/code/Bits2Bites/dataset_prep

# BracketPrediction is vendored in-repo at dataset_prep/BracketPrediction/
# (branch new_landmarks, own nested .git). Override via env if testing an
# alternate checkout elsewhere.
BRACKET_DIR=${BRACKET_DIR:-$PREP_DIR/BracketPrediction}
cd "$BRACKET_DIR"
source "$PREP_DIR/.venv/bin/activate"

LM_IN=${LM_IN:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_in}
LM_OUT=${LM_OUT:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_out}

# Configs are in-repo; weights are the LOCAL copies under dataset_prep/weights,
# so this needs no access to anyone else's tree.
# - SEG_WEIGHT: unchanged, originally sourced from
#   Mlugli/models/AutoBonding/segmentator_best.pth.
# - BOND_WEIGHT: heatmap_landmarks.pth, sourced from
#   /work/grana_maxillo/averonese_STS2026/BracketPrediction/checkpoints/
#   (newest checkpoint as of 2026-07-01; supersedes the old
#   Mlugli/models/Brackets/3dteethland_0/model/model_best.pth-derived
#   bond_model_best.pth, which is now stale).
SEG_CONFIG=${SEG_CONFIG:-application/app_configs/Pt_semseg_teeth3ds_app.py}
SEG_WEIGHT=${SEG_WEIGHT:-$PREP_DIR/weights/segmentator_best.pth}
BOND_CONFIG=${BOND_CONFIG:-application/app_configs/Pt_landmarks_app.py}
BOND_WEIGHT=${BOND_WEIGHT:-$PREP_DIR/weights/heatmap_landmarks.pth}
PREPROCESSING=${PREPROCESSING:-3dteethland_preprocessing.yaml}

python main.py \
    --samples "$LM_IN/testing_lower.txt" "$LM_IN/testing_upper.txt" \
    --data-folder "$LM_IN" \
    --output-folder "$LM_OUT" \
    --seg-config  "$SEG_CONFIG" \
    --seg-weight  "$SEG_WEIGHT" \
    --bond-config "$BOND_CONFIG" \
    --bond-weight "$BOND_WEIGHT" \
    --preprocessing "$PREPROCESSING" \
    --cache --save-ply

echo "predictions.csv written under $LM_OUT/<timestamp>_<hex>/ — ingest with 05_landmarks_ingest.py"
