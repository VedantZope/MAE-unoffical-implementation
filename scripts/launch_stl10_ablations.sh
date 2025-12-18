#!/usr/bin/env bash
set -euo pipefail

# Launches the STL-10 MAE ablation grid (6 runs) in parallel, one per GPU.
# Pretraining on STL-10 uses train+unlabeled split (handled in src/data/stl10.py).
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
  "configs/mae_vit_tiny_stl10_mask50_dec2.yaml"
  "configs/mae_vit_tiny_stl10_mask75_dec2.yaml"
  "configs/mae_vit_tiny_stl10_mask90_dec2.yaml"
  "configs/mae_vit_tiny_stl10_mask50_dec4.yaml"
  "configs/mae_vit_tiny_stl10_mask75_dec4.yaml"
  "configs/mae_vit_tiny_stl10_mask90_dec4.yaml"
)

if [[ "${#CONFIGS[@]}" -gt 8 ]]; then
  echo "This script expects <= 8 configs (one per GPU)."
  exit 1
fi

mkdir -p experiments/logs

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  log="experiments/logs/$(basename "$cfg" .yaml).log"
  echo "GPU $i -> $cfg (log: $log)"

  (
    export CUDA_VISIBLE_DEVICES="$i"
    conda run -n mae-compact python -u -m src.train.train_mae --config "$cfg" --log-interval 10
  ) >"$log" 2>&1 &
done

echo "Launched ${#CONFIGS[@]} runs."
echo "Tail a run with: tail -f experiments/logs/<config>.log"
wait
