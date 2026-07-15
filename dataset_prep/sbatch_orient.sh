#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --constraint="gpu_L40S_45G"
#SBATCH --account=grana_maxillo
#SBATCH --gres=gpu:1
#SBATCH --job-name=b2b_orient
#SBATCH --output=b2b_orient_%j.out
#SBATCH --error=b2b_orient_%j.err
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

set -euo pipefail

PREP_DIR=/work/grana_maxillo/lborghi/code/Bits2Bites/dataset_prep
cd "$PREP_DIR"

source "$PREP_DIR/.venv/bin/activate"

MERGED=${MERGED:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_merged}
ORIENTED=${ORIENTED:-/work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented}

CKPT=${CKPT:-$PREP_DIR/weights/orient_pointnet_best.pt}

python 03_orient.py \
    --dataset "$MERGED" \
    --output "$ORIENTED" \
    --checkpoint "$CKPT" \
    --qa 10 \
    --force
