#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=grana_maxillo
#SBATCH --gres=gpu:1
#SBATCH --job-name=dental_ptv3_mtl_fold1_mesh
#SBATCH --output=dental_ptv3_mtl_fold1_mesh_%j.out
#SBATCH --error=dental_ptv3_mtl_fold1_mesh_%j.err
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

# wandb auth: WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT are exported from
# /work/grana_maxillo/lborghi/env.sh (sourced above, not git-tracked).
# See BITS2BITES.md "Weights & Biases (wandb) setup" for how to configure it.

# Mesh-only variant (BITS2BITES.md "Input" section): strips the 6 landmark
# classes from the already-built data/dental_landmarks_mesh, keeping only
# "Mesh" points (~198k/sample, same as the full variant) -> similar runtime
# to the full mesh+landmarks run, but with an all-zero landmark one-hot.
# python pointcept/datasets/preprocessing/dental/filter_mesh_only.py \
#     --input-dir data/dental_landmarks_mesh \
#     --output-dir data/dental_mesh_only

python tools/dental_fold.py --fold-val 1 --data-root data/dental_mesh_only
sh scripts/train.sh -d dental -c cls-ptv3-base -n ptv3_mtl_fold1_mesh -g 1 -e data/dental_mesh_only
sh scripts/test.sh  -d dental -n ptv3_mtl_fold1_mesh -g 1
# metrics -> exp/dental/ptv3_mtl_fold1_mesh/result/metrics.json
