#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=grana_maxillo
#SBATCH --gres=gpu:1
#SBATCH --job-name=dental_ptv3_mtl_fold1
#SBATCH --output=dental_ptv3_mtl_fold1_%j.out
#SBATCH --error=dental_ptv3_mtl_fold1_%j.err
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

# Dataset already prepared at data/dental_landmarks_mesh (see BITS2BITES.md).
python tools/dental_fold.py --fold-val 1 --data-root data/dental_landmarks_mesh
sh scripts/train.sh -d dental -c cls-ptv3-base -n ptv3_mtl_fold1 -g 1
sh scripts/test.sh  -d dental -n ptv3_mtl_fold1 -g 1
# metrics -> exp/dental/ptv3_mtl_fold1/result/metrics.json
