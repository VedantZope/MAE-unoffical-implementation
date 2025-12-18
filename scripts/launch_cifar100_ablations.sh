#!/usr/bin/env bash
set -euo pipefail

# Launches the CIFAR-100 MAE ablation grid (6 runs) in parallel, one per GPU.
# Assumes:
#   - conda env `mae-compact` exists
#   - CIFAR-100 already downloaded in `data/`
#   - you want NO cloud sync (offline W&B logs saved locally under ./wandb/)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
# Avoid CPU thread oversubscription when running many jobs in parallel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CONFIGS=(
  "configs/pretraining/cifar100/mae_vit_tiny_cifar100_mask50_dec2.yaml"
  "configs/pretraining/cifar100/mae_vit_tiny_cifar100_mask75_dec2.yaml"
  "configs/pretraining/cifar100/mae_vit_tiny_cifar100_mask90_dec2.yaml"
  "configs/pretraining/cifar100/mae_vit_tiny_cifar100_mask50_dec4.yaml"
  "configs/pretraining/cifar100/mae_vit_tiny_cifar100_mask75_dec4.yaml"
  "configs/pretraining/cifar100/mae_vit_tiny_cifar100_mask90_dec4.yaml"
)

# GPU mapping:
# - Default: use 6 GPUs and skip GPU 2 (often flaky on some rigs).
# - Override by setting `GPU_IDS`, e.g.:
#     GPU_IDS="0,1,3,4,5,6" bash scripts/launch_cifar100_ablations.sh
#     GPU_IDS="0 1 3 4 5 6" bash scripts/launch_cifar100_ablations.sh
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
  echo "Set GPU_IDS, e.g. GPU_IDS=\"0,1,3,4,5,6\""
  exit 1
fi

mkdir -p experiments/logs

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  gpu="${GPU_ID_LIST[$i]}"
  log="experiments/logs/$(basename "$cfg" .yaml).log"
  echo "GPU $gpu -> $cfg (log: $log)"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u -m src.train.train_mae --config "$cfg" --log-interval 10
  ) >"$log" 2>&1 &
done

echo "Launched ${#CONFIGS[@]} runs."
echo "Tail a run with: tail -f experiments/logs/<config>.log"
wait
