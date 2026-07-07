#!/bin/bash
# Create the shared dataset_prep virtualenv with uv.
#
# This is SEPARATE from the main Bits2Bites .venv (code/Bits2Bites/.venv) and is
# only used by the CPU/GPU dataset-prep scripts here (01..05). It never touches
# the main training env.
#
# Two consumers share this single venv:
#  1. ios_orienter's PointNet regressor (03_orient.py) — needs numpy+torch,
#     plus trimesh+matplotlib for STL export and QA montages. No strict torch
#     version pin.
#  2. BracketPrediction's 3DTeethLand landmark predictor (sbatch_3dteethland.sh,
#     vendored at dataset_prep/BracketPrediction/) — needs torch 2.5.0+cu124 to
#     match the spconv-cu124 / torch-scatter / torch-cluster / torch-geometric
#     wheels below (those are only published for specific torch+cuda combos),
#     plus its own pointops CUDA extension built from libs/pointops.
#
# NOTE: bumped from torch 2.5.1+cu121 to 2.5.0+cu124 to satisfy (2). ios_orienter
# has no pin so this is expected to be safe, but re-run 03_orient.py once after
# this change as a sanity check (compare against a known-good prior orientation
# output) before trusting it blindly.
set -euo pipefail

PREP_DIR=/work/grana_maxillo/lborghi/code/Bits2Bites/dataset_prep
BRACKET_DIR="$PREP_DIR/BracketPrediction"
cd "$PREP_DIR"

uv venv --python 3.10 .venv
source .venv/bin/activate

# --- torch (cu124), matching torchvision and the pyg-family wheels below ---
uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.0+cu124 torchvision==0.20.0+cu124

# --- ios_orienter deps ---
uv pip install numpy trimesh matplotlib

# --- BracketPrediction inference-only deps ---
# (traced from the actual runtime import graph of main.py; deliberately
# excludes train-only / guarded / unreachable deps such as flash-attn, CLIP,
# ocnn-pytorch, wandb, tensorboard(X), black, open3d, pointgroup_ops, pointops2,
# dwconv, ftfy, regex, torch-sparse, google-sparsehash, h5py, ninja, rtree,
# pandas, sympy, numpy-stl — see PIPELINE.md for the full reasoning)
uv pip install spconv-cu124
uv pip install --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html \
    torch-scatter torch-cluster torch-geometric
uv pip install einops timm addict yapf termcolor scipy scikit-learn \
    PyYAML pyvista plyfile meshlib faiss-cpu debugpy

# --- BracketPrediction's pointops CUDA extension ---
# Builds a torch.utils.cpp_extension.CUDAExtension: needs nvcc matching the
# installed torch's CUDA version (12.4) plus a C/C++ toolchain. The login node
# has no GPU (see PIPELINE.md) and may not have nvcc/the CUDA toolkit module
# loaded either — VERIFY this builds here; if it fails, load the CUDA 12.4
# module first, or run this install step from an interactive/one-off sbatch
# GPU session instead (`srun --partition=all_usr_prod --account=grana_maxillo
# --gres=gpu:1 --pty bash`, then re-run just this `uv pip install` line with
# the venv activated).

uv pip install "$BRACKET_DIR/libs/pointops" --no-build-isolation

echo "dataset_prep venv ready at $PREP_DIR/.venv"
python -c "import torch, trimesh, matplotlib, spconv, torch_scatter, torch_cluster, torch_geometric, pointops; print('imports OK')"
