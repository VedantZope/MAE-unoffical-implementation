#!/usr/bin/env bash
set -euo pipefail

# Launches the Tiny-ImageNet MAE ablation grid (6 runs) in parallel, one per GPU.
# Dataset is NOT auto-downloaded; expects `data/tiny-imagenet-200/` to exist.
# W&B is forced offline (local logs under ./wandb/).

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
  "configs/pretraining/tinyimagenet/mae_vit_tiny_tinyimagenet_mask50_dec2.yaml"
  "configs/pretraining/tinyimagenet/mae_vit_tiny_tinyimagenet_mask75_dec2.yaml"
  "configs/pretraining/tinyimagenet/mae_vit_tiny_tinyimagenet_mask90_dec2.yaml"
  "configs/pretraining/tinyimagenet/mae_vit_tiny_tinyimagenet_mask50_dec4.yaml"
  "configs/pretraining/tinyimagenet/mae_vit_tiny_tinyimagenet_mask75_dec4.yaml"
  "configs/pretraining/tinyimagenet/mae_vit_tiny_tinyimagenet_mask90_dec4.yaml"
)

mkdir -p experiments/logs

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  log="experiments/logs/$(basename "$cfg" .yaml).log"
  echo "GPU $i -> $cfg (log: $log)"

  (
    export CUDA_VISIBLE_DEVICES="$i"
    python -u -m src.train.train_mae --config "$cfg" --log-interval 10
  ) >"$log" 2>&1 &
done

echo "Launched ${#CONFIGS[@]} runs."
echo "Tail a run with: tail -f experiments/logs/<config>.log"
wait
