# MAE-unoffical-implementation


example project structure:
```
mae-project/
│
├── README.md
├── requirements.txt
├── setup.py                     # optional
├── .gitignore
│
├── configs/
│   ├── mae_vit_tiny_tinyimagenet_mask75_dec2.yaml
│   ├── mae_vit_tiny_mask50.yaml
│   ├── mae_vit_tiny_mask90.yaml
│   ├── mae_decoder_depth2.yaml
│   ├── mae_decoder_depth4.yaml
│   └── linear_probe_cifar100.yaml
│
├── data/
│   └── (empty; user downloads datasets here)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cifar100.py
│   │   ├── stl10.py
│   │   ├── tiny_imagenet.py
│   │   └── factory.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patch_embed.py
│   │   ├── vit_encoder.py
│   │   ├── masking.py
│   │   ├── mae_decoder.py
│   │   ├── mae.py                # MAE wrapper that ties encoder+decoder
│   │   └── linear_probe_head.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   └── mae_loss.py
│   │
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_mae.py          # pretraining script
│   │   ├── train_linear_probe.py
│   │   └── train_finetune.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   ├── seed.py
│   │   └── distributed.py        # optional, if using multi-GPU
│   │
│   └── eval/
│       ├── __init__.py
│       ├── run_linear_probe.py
│       └── run_finetune_eval.py
│
├── experiments/
│   ├── logs/                     # training logs, wandb/tb outputs
│   ├── mae_pretrained/           # encoder+decoder checkpoints
│   ├── linear_probe/             # trained linear classifiers
│   └── finetune/                 # fine-tuned models
│
├── analysis/
│   ├── mae_reconstruction_demo.ipynb
│   ├── ablation_plots.ipynb
│   └── metrics.csv               # accuracy table for report
│
└── report/
    ├── final_report.pdf
    ├── figures/
    │   ├── recon_examples.png
    │   ├── mask_ratio_plot.png
    │   └── decoder_depth_plot.png
    └── notes.md
```