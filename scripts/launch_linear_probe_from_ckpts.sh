#!/usr/bin/env bash
set -euo pipefail

# Example launcher to run linear probing for a list of MAE checkpoints.
# Edit CKPTS below to point at your finished pretraining checkpoints.
#
# Uses offline W&B (local runs under ./wandb/).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

mkdir -p experiments/logs

DATASET="${1:-cifar100}" # cifar100 | stl10
# Update these paths to your actual checkpoints (or keep the defaults).
if [[ "${DATASET}" == "stl10" ]]; then
  CKPTS=(
    "experiments/mae_pretrained/mae_stl10_vit_tiny_img224_p16_mask50_dec2.pth"
    "experiments/mae_pretrained/mae_stl10_vit_tiny_img224_p16_mask75_dec2.pth"
    "experiments/mae_pretrained/mae_stl10_vit_tiny_img224_p16_mask90_dec2.pth"
    "experiments/mae_pretrained/mae_stl10_vit_tiny_img224_p16_mask50_dec4.pth"
    "experiments/mae_pretrained/mae_stl10_vit_tiny_img224_p16_mask75_dec4.pth"
    "experiments/mae_pretrained/mae_stl10_vit_tiny_img224_p16_mask90_dec4.pth"
  )
else
  CKPTS=(
    "experiments/mae_pretrained/mae_cifar100_vit_tiny_img224_p16_mask50_dec2.pth"
    "experiments/mae_pretrained/mae_cifar100_vit_tiny_img224_p16_mask75_dec2.pth"
    "experiments/mae_pretrained/mae_cifar100_vit_tiny_img224_p16_mask90_dec2.pth"
    "experiments/mae_pretrained/mae_cifar100_vit_tiny_img224_p16_mask50_dec4.pth"
    "experiments/mae_pretrained/mae_cifar100_vit_tiny_img224_p16_mask75_dec4.pth"
    "experiments/mae_pretrained/mae_cifar100_vit_tiny_img224_p16_mask90_dec4.pth"
  )
fi

# GPU mapping:
# - Default: use 6 GPUs and skip GPU 2 (often flaky on some rigs).
# - Override by setting `GPU_IDS`, e.g.:
#     GPU_IDS="0,1,3,4,5,6" bash scripts/launch_linear_probe_from_ckpts.sh cifar100
#     GPU_IDS="0 1 3 4 5 6" bash scripts/launch_linear_probe_from_ckpts.sh stl10
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

if [[ "${#GPU_ID_LIST[@]}" -lt "${#CKPTS[@]}" ]]; then
  echo "Not enough GPUs listed in GPU_IDS (${#GPU_ID_LIST[@]}) for ${#CKPTS[@]} checkpoints."
  echo "Set GPU_IDS, e.g. GPU_IDS=\"0,1,3,4,5,6\""
  exit 1
fi

for i in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$i]}"
  gpu="${GPU_ID_LIST[$i]}"

  if [[ ! -f "$ckpt" ]]; then
    echo "[WARN] Missing ckpt: $ckpt (skipping)"
    continue
  fi

  base="$(basename "$ckpt" .pth)"
  cfg="configs/_tmp_lprobe_${base}.yaml"
  log="experiments/logs/${base}_lprobe_${DATASET}.log"

  cat >"$cfg" <<EOF
dataset: ${DATASET}
data_root: data
img_size: 224
eval_split: test
ckpt_path: ${ckpt}
patch_size: 16
embed_dim: 192
encoder_depth: 12
encoder_heads: 3
encoder_mlp_ratio: 4.0
dropout: 0.0
pool: cls
batch_size: 256
epochs: 50
lr: 0.1
momentum: 0.9
weight_decay: 0.0
num_workers: 8
output_dir: experiments/linear_probe
run_name: lprobe_${DATASET}_${base}
use_wandb: true
wandb_project: mae-compact
wandb_run_name: lprobe_${DATASET}_${base}
wandb_tags: ["linear_probe", "${DATASET}"]
wandb_mode: offline
EOF

  echo "GPU ${gpu} -> linear probe ${DATASET} on ${ckpt} (log: ${log})"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python -u -m src.train.train_linear_probe --config "$cfg"
  ) >"$log" 2>&1 &
done

echo "Launched linear probing for ${DATASET} (${#CKPTS[@]} runs)."
echo "Tail a run with: tail -f experiments/logs/<ckpt>_lprobe_${DATASET}.log"
wait
