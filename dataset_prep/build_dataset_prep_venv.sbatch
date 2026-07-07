#!/bin/bash
#SBATCH --job-name=build_dataset_prep_venv
#SBATCH --partition=boost_usr_prod
#SBATCH --account=grana_maxillo
#SBATCH --gpus=1
#SBATCH --constraint="gpu_L40S_45G"
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PREP_DIR=/work/grana_maxillo/lborghi/code/Bits2Bites/dataset_prep
BRACKET_DIR="$PREP_DIR/BracketPrediction"

cd "$PREP_DIR"

echo "== Node =="
hostname
nvidia-smi

echo "== Modules =="
module purge
module load cuda/12.6.3

which nvcc
nvcc --version

echo "== Recreate venv from scratch =="
rm -rf .venv
uv venv --python 3.10 .venv
source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel

echo "== Install torch cu124 =="
uv pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.version.cuda != "12.4":
    raise SystemExit(f"Expected torch CUDA 12.4, got {torch.version.cuda}")
PY

echo "== ios_orienter deps =="
uv pip install numpy trimesh matplotlib

echo "== BracketPrediction deps =="
uv pip install spconv-cu124 wandb

uv pip install --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html \
    torch-scatter torch-cluster torch-geometric

uv pip install einops timm addict yapf termcolor scipy scikit-learn \
    PyYAML pyvista plyfile meshlib faiss-cpu debugpy

echo "== Force torch back to cu124 before pointops =="
uv pip install --reinstall torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124

echo "== Build pointops =="
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=$CUDA_HOME"

# Optional but often helps avoid compiling for every architecture.
# L40S is Ada Lovelace, compute capability 8.9.
export TORCH_CUDA_ARCH_LIST="8.9"

uv pip install "$BRACKET_DIR/libs/pointops" --no-build-isolation -v

echo "== Final import test =="
python - <<'PY'
import torch
import trimesh
import matplotlib
import spconv
import torch_scatter
import torch_cluster
import torch_geometric
import pointops

print("imports OK")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY

echo "dataset_prep venv ready at $PREP_DIR/.venv"
