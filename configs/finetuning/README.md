# Fine-tuning configs

These YAMLs fine-tune a classification head + the full ViT encoder end-to-end starting from a pretrained MAE checkpoint.

Run:

```bash
python -m src.train.train_finetune --config <path/to/config.yaml>
```

Notes:
- Uses `train_augment: true` by default (RandomResizedCrop + flip), while evaluation uses center-crop.
- `ckpt_path` should point to a MAE pretraining checkpoint in `experiments/mae_pretrained/`.
- Separate learning rates are used for encoder vs head via `encoder_lr_scale`.

