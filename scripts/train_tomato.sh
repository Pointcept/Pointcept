#!/bin/bash
# Script to train Point Transformer V3 on Tomato dataset

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate pointcept environment
conda activate pointcept

# Check environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

export CUDA_VISIBLE_DEVICES=0,1  # 使用两块 GPU

# Add Pointcept to Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Number of GPUs to use
NUM_GPUS=2

# Config file
CONFIG="configs/tomato/semseg-pt-v3m1-0-tomato.py"

# 预训练权重（从 ScanNet 转换而来）
PRETRAINED_WEIGHT="model_best_for_tomato.pth"

# Experiment name (will create exp folder with this name)
EXP_NAME="ptv3m1_tomato20480_finetune_$(date +%Y%m%d_%H%M%S)"

# Training command
python tools/train.py \
    --config-file ${CONFIG} \
    --num-gpus ${NUM_GPUS} \
    --options save_path=exp/tomato/${EXP_NAME} weight=${PRETRAINED_WEIGHT}

echo "Training finished! Results saved to exp/tomato/${EXP_NAME}"
