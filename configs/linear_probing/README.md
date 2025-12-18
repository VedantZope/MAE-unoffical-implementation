# Linear probing configs

These YAMLs are meant to be run with:

`python -u -m src.train.train_linear_probe --config <yaml>`

They assume the corresponding MAE checkpoints already exist under `experiments/mae_pretrained/`.

Examples:
- `python -u -m src.train.train_linear_probe --config configs/linear_probing/cifar100/lprobe_cifar100_mask75_dec2.yaml`
- `python -u -m src.train.train_linear_probe --config configs/linear_probing/stl10/lprobe_stl10_mask75_dec2.yaml`

