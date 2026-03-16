#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "pointcept" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="conda"
  elif [ -x "/home/hwh/anaconda3/bin/conda" ]; then
    CONDA_BIN="/home/hwh/anaconda3/bin/conda"
  else
    echo "conda not found. Please ensure conda is installed and on PATH." >&2
    exit 1
  fi

  # Some conda activate/deactivate hooks are not 'set -u' safe (e.g. ZSH_VERSION).
  set +u
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate pointcept
  set -u
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

NUM_GPUS="${NUM_GPUS:-1}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-exp/tomato_gs2pc/ablations/${TAG}}"
SEEDS_STR="${SEEDS:-0}"

mkdir -p "$OUT_ROOT"

configs=(
  "baseline:configs/tomato_gs2pc/semseg-spunet-v1m1-0-tomato-gs2pc-200k.py"
  "ms:configs/tomato_gs2pc/semseg-spunet-v1m1-3-tomato-gs2pc-200k-ms.py"
  "skipgate:configs/tomato_gs2pc/semseg-spunet-v1m1-4-tomato-gs2pc-200k-skipgate.py"
  "ms_skipgate_se:configs/tomato_gs2pc/semseg-spunet-v1m1-5-tomato-gs2pc-200k-ms-skipgate-se.py"
  "boundary:configs/tomato_gs2pc/semseg-spunet-v1m1-6-tomato-gs2pc-200k-boundary.py"
  "organ_expert_fused:configs/tomato_gs2pc/semseg-spunet-v1m1-2-tomato-gs2pc-200k-organ-expert-fused.py"
  "organ_expert_fused_boundary:configs/tomato_gs2pc/semseg-spunet-v1m1-7-tomato-gs2pc-200k-organ-expert-fused-boundary.py"
)

echo "Output root: $OUT_ROOT"
echo "Seeds: $SEEDS_STR"
echo "Num GPUs: $NUM_GPUS"

for seed in $SEEDS_STR; do
  for item in "${configs[@]}"; do
    name="${item%%:*}"
    cfg="${item#*:}"
    save_path="${OUT_ROOT}/${name}_seed${seed}"
    if [[ "${FORCE:-0}" != "1" && -f "${save_path}/train.log" ]]; then
      echo "=== Skipping existing: ${name} seed=${seed} (set FORCE=1 to rerun) ==="
      continue
    fi
    echo "=== Running: ${name} seed=${seed} -> ${save_path} ==="
    python tools/train.py \
      --config-file "$cfg" \
      --num-gpus "$NUM_GPUS" \
      --options save_path="$save_path" seed="$seed"
  done
done

echo "All done. Summarize with:"
echo "  python scripts/summarize_ablation.py --root ${OUT_ROOT}"
