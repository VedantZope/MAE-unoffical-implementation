#!/usr/bin/env bash
set -euo pipefail

# Launch a small set of fine-tuning runs (end-to-end) from the best MAE checkpoints.
# Assumes you already activated your conda env (e.g. `conda activate mae-compact`).
#
# Default GPU list skips GPU 2 (often flaky on some rigs).
#
# Usage:
#   bash scripts/launch_finetune_best.sh
#
# Optional:
#   GPU_IDS="0,1,3,4" bash scripts/launch_finetune_best.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CONFIGS=(
  "configs/finetuning/cifar100/finetune_cifar100_mask90_dec2.yaml"
  "configs/finetuning/cifar100/finetune_cifar100_mask90_dec4.yaml"
  "configs/finetuning/stl10/finetune_stl10_mask90_dec4.yaml"
  "configs/finetuning/stl10/finetune_stl10_mask75_dec2.yaml"
)

GPU_IDS_DEFAULT=(0 1 3 4 5 6)
GPU_IDS_STR="${GPU_IDS:-}"
GPU_ID_LIST=()
if [[ -n "${GPU_IDS_STR}" ]]; then
  GPU_IDS_CLEAN="${GPU_IDS_STR//,/ }"
  # shellcheck disable=SC2206
  GPU_ID_LIST=(${GPU_IDS_CLEAN})
else
  GPU_ID_LIST=("${GPU_IDS_DEFAULT[@]}")
fi

if [[ "${#GPU_ID_LIST[@]}" -lt "${#CONFIGS[@]}" ]]; then
  echo "Not enough GPUs listed in GPU_IDS (${#GPU_ID_LIST[@]}) for ${#CONFIGS[@]} configs."
  echo "Set GPU_IDS, e.g. GPU_IDS=\"0,1,3,4\""
  exit 1
fi

mkdir -p experiments/logs

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  gpu="${GPU_ID_LIST[$i]}"
  log="experiments/logs/$(basename "$cfg" .yaml).log"

  echo "GPU ${gpu} -> finetune $cfg (log: $log)"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python -u -m src.train.train_finetune --config "$cfg"
  ) >"$log" 2>&1 &
done

echo "Launched ${#CONFIGS[@]} fine-tuning runs."
echo "Tail: tail -f experiments/logs/<config>.log"
wait

