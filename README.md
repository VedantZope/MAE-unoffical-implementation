# Compact MAE Reproduction (PyTorch)

This repo is a compact reproduction of *Masked Autoencoders Are Scalable Vision Learners* (He et al., 2021).
We implement MAE pretraining and evaluate representation quality via:
- **Linear probing** (freeze encoder, train linear head)
- **Fine-tuning** (train encoder + head end-to-end)

We run controlled ablations over:
- **Mask ratio**: `{0.50, 0.75, 0.90}`
- **Decoder depth**: `{2, 4}` (decoder width set to ~half encoder width)

Experiments in this repo target **CIFAR-100** and **STL-10** (with optional Tiny-ImageNet support).

---

## Quickstart

### 1) Environment
We provide conda environments:
- `environment-cuda.yml` (GPU)
- `environment-cpu.yml` (CPU)

Example:
```bash
conda env create -f environment-cuda.yml
conda activate mae-compact
```

### 2) Datasets
All datasets are stored under `data/`.
- CIFAR-100: auto-download (via torchvision)
- STL-10: auto-download (via torchvision; MAE pretraining uses `train+unlabeled`)


### 3) Run the full pipeline
All scripts assume you already activated the conda env and run `python` directly.
Logging:
- stdout/stderr logs: `experiments/logs/*.log`
- checkpoints: `experiments/mae_pretrained/`, `experiments/linear_probe/`, `experiments/finetune/`
- W&B is forced to **offline** by default in launchers (stored under `./wandb/`)

---

## Repo Layout (current)
Key directories:
```
configs/
  pretraining/{cifar100,stl10}/...yaml
  linear_probing/{cifar100,stl10}/...yaml
  finetuning/{cifar100,stl10}/...yaml
scripts/
  launch_*_ablations.sh
  launch_linear_probe_from_ckpts.sh
  launch_finetune_best.sh
  launch_finetune_remaining.sh
src/
  train/train_mae.py
  train/train_linear_probe.py
  train/train_finetune.py
  models/{vit_encoder.py,mae.py,mae_decoder.py,masking.py}
  data/{factory.py,cifar100.py,stl10.py,tiny_imagenet.py}
notebooks/
  reconstruction_demo.ipynb
  analysis_notebook.ipynb
experiments/
  logs/            # logs for each run
  mae_pretrained/  # MAE checkpoints
  linear_probe/    # trained linear heads
  finetune/        # fine-tuned models
```

---

## Pretraining (MAE)
Run a single config:
```bash
python -u -m src.train.train_mae --config configs/pretraining/cifar100/mae_vit_tiny_cifar100_mask75_dec2.yaml
```

Outputs:
- checkpoints: `experiments/mae_pretrained/*.pth`
- logs: `experiments/logs/mae_vit_tiny_*_mask*_dec*.log`

---

## Linear Probing
Run a single YAML:
```bash
python -u -m src.train.train_linear_probe --config configs/linear_probing/cifar100/lprobe_cifar100_mask75_dec2.yaml
```

Outputs:
- logs: `experiments/logs/*_lprobe_*.log`
- best linear heads: `experiments/linear_probe/*.pth`

---

## Fine-tuning
Run a single YAML:
```bash
python -u -m src.train.train_finetune --config configs/finetuning/cifar100/finetune_cifar100_mask75_dec4.yaml
```

Outputs:
- logs: `experiments/logs/finetune_*.log`
- checkpoints: `experiments/finetune/*.pth`

---

## Qualitative Reconstructions
Use `notebooks/reconstruction_demo.ipynb` to visualize:
- original image
- masked input
- **composite reconstruction** (visible patches from input + masked patches from model)

---

## Testing
Run unit tests in the provided conda env:
```bash
conda run -n mae-compact python -m pytest -q
```

---

## Notes for Graders (TA/Instructor)
- All experiments are driven by YAML configs under `configs/`.
- The canonical outputs are in `experiments/` (logs + checkpoints).
- The model weights can be found here: [Drive link](<drive-link>)
