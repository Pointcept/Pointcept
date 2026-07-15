#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=grana_maxillo
#SBATCH --gres=gpu:1
#SBATCH --job-name=dental_ptv3_mtl_fold1_landmarks_v2
#SBATCH --output=dental_ptv3_mtl_fold1_landmarks_v2_%j.out
#SBATCH --error=dental_ptv3_mtl_fold1_landmarks_v2_%j.err
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=10:00:00

# --constraint="gpu_RTX5000_16G|gpu_RTXA5000_24G|gpu_RTX6000_24G|gpu_2080Ti_11G"

set -euo pipefail

REPO_DIR=/work/grana_maxillo/lborghi/code/Bits2Bites
cd "$REPO_DIR"

source /work/grana_maxillo/lborghi/env.sh
source "$REPO_DIR/.venv/bin/activate"

# uv sync
# uv sync --extra gpu
# uv pip install ./libs/pointops --no-build-isolation

# wandb auth: WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT are exported from
# /work/grana_maxillo/lborghi/env.sh (sourced above, not git-tracked).
# See BITS2BITES.md "Weights & Biases (wandb) setup" for how to configure it.

# Landmark-only variant (BITS2BITES.md "Input" section): strips "Mesh"
# points from the already-built data/dental_landmarks_mesh, ~240 pts/sample
# instead of ~198k -> much faster, same configs apply.
# python pointcept/datasets/preprocessing/dental/filter_landmarks_only.py \
#     --input-dir data/dental_landmarks_mesh \
#     --output-dir data/dental_landmarks_only

python tools/dental_fold.py --fold-val 1 --data-root data/dental_landmarks_only_v2
sh scripts/train.sh -d dental_v2 -c cls-ptv3-landmarks-v2 -n ptv3_mtl_fold1_landmarks -g 1 -e data/dental_landmarks_only_v2
sh scripts/test.sh  -d dental_v2 -n ptv3_mtl_fold1_landmarks -g 1
# metrics -> exp/dental_v2/ptv3_mtl_fold1_landmarks/result/metrics.json
