import argparse
import math
import os
from types import SimpleNamespace
from typing import Dict, Any, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.factory import get_dataloader
from src.losses.mae_loss import MAELoss
from src.models.mae import MAEModel
from src.models.mae_decoder import MAEDecoder
from src.models.vit_encoder import ViTEncoder


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert images (B, 3, H, W) to patch tokens (B, num_patches, patch_dim).
    Expects H and W to be divisible by patch_size.
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dims must be divisible by patch size"
    h = H // patch_size
    w = W // patch_size
    # (B, C, h, patch, w, patch) -> (B, h*w, patch*patch*C)
    imgs = imgs.reshape(B, C, h, patch_size, w, patch_size)
    imgs = imgs.permute(0, 2, 4, 3, 5, 1)
    patches = imgs.reshape(B, h * w, patch_size * patch_size * C)
    return patches


def build_model(args) -> nn.Module:
    encoder = ViTEncoder(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.encoder_depth,
        num_heads=args.encoder_heads,
        mlp_ratio=args.encoder_mlp_ratio,
        dropout=args.dropout,
    )
    decoder = MAEDecoder(
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        num_patches=(args.img_size // args.patch_size) ** 2,
        patch_size=args.patch_size,
        depth=args.decoder_depth,
        num_heads=args.decoder_heads,
        mlp_ratio=args.decoder_mlp_ratio,
        dropout=args.dropout,
    )
    return MAEModel(encoder=encoder, decoder=decoder, mask_ratio=args.mask_ratio)


def maybe_init_wandb(cfg_dict: Dict[str, Any]):
    if not cfg_dict.get("use_wandb", False):
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[WARN] use_wandb=True but wandb import failed: {e}")
        print("[WARN] Proceeding without wandb logging.")
        return None

    # Allow environment override (useful when configs default to offline).
    # Examples:
    #   export WANDB_MODE=online
    #   export WANDB_MODE=offline
    env_mode = os.environ.get("WANDB_MODE")
    mode = env_mode if env_mode is not None else cfg_dict.get("wandb_mode", None)

    return wandb.init(
        project=cfg_dict.get("wandb_project", "mae-compact"),
        entity=cfg_dict.get("wandb_entity", None),
        name=cfg_dict.get("wandb_run_name", None),
        tags=cfg_dict.get("wandb_tags", None),
        mode=mode,  # "online" | "offline" | "disabled" | None
        config=cfg_dict,
    )


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
    base_lr: Optional[float] = None,
):
    if base_lr is None:
        base_lr = float(optimizer.param_groups[0]["lr"])

    total_steps = max(int(epochs * steps_per_epoch), 1)
    warmup_steps = int(max(warmup_epochs, 0) * steps_per_epoch)
    warmup_steps = min(warmup_steps, total_steps)

    min_lr_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps == warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    ckpt_path: str,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    patch_size: int,
    log_interval: int,
    scheduler=None,
    max_steps_per_epoch: Optional[int] = None,
    wandb_run=None,
    global_step: int = 0,
    amp: bool = False,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    steps_done = 0

    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    optimizer.zero_grad(set_to_none=True)

    for step, imgs in enumerate(dataloader):
        if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
            break
        imgs = imgs.to(device, non_blocking=True)
        target = patchify(imgs, patch_size)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            pred, mask = model(imgs)
            loss = criterion(pred, target, mask)

        loss_to_backprop = loss / float(grad_accum_steps)
        if amp and device.type == "cuda":
            assert scaler is not None
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        total_loss += loss.item()
        steps_done = step + 1

        do_step = (steps_done % grad_accum_steps == 0)
        if do_step:
            if amp and device.type == "cuda":
                assert scaler is not None
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        if (steps_done % log_interval) == 0:
            avg_loss = total_loss / steps_done
            lr = optimizer.param_groups[0]["lr"]
            print(f"Step [{steps_done}/{len(dataloader)}]  loss={loss.item():.4f}  avg_loss={avg_loss:.4f}  lr={lr:.6f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/loss_avg": avg_loss,
                        "train/lr": lr,
                        "train/step": global_step + steps_done,
                    }
                )

    return total_loss / max(int(steps_done), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train MAE (YAML-configured)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Optional override for device in YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for output_dir in YAML")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for seed in YAML")
    parser.add_argument("--log-interval", type=int, default=None, help="Optional override for log_interval in YAML")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyYAML is required to run this script. Install it with `pip install pyyaml`."
        ) from e

    # defaults mirror prior CLI defaults
    default_cfg: Dict[str, Any] = {
        "dataset": "cifar100",
        "data_root": "data",
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-4,
        "min_lr": 0.0,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "mask_ratio": 0.75,
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 768,
        "encoder_depth": 12,
        "encoder_heads": 12,
        "encoder_mlp_ratio": 4.0,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 8,
        "decoder_mlp_ratio": 4.0,
        "dropout": 0.0,
        "num_workers": 4,
        "log_interval": 50,
        "max_steps_per_epoch": None,
        "output_dir": "experiments/mae_pretrained",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "norm_pix_loss": False,
        "seed": 42,
        "amp": True,
        "grad_accum_steps": 1,
        "use_wandb": False,
        "wandb_project": "mae-compact",
        "wandb_entity": None,
        "wandb_run_name": None,
        "wandb_tags": None,
        "wandb_mode": None,  # "online" | "offline" | "disabled" | None
    }

    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    # merge YAML into defaults
    cfg_dict = {**default_cfg, **yaml_cfg}

    # CLI overrides
    if args.device is not None:
        cfg_dict["device"] = args.device
    if args.output_dir is not None:
        cfg_dict["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    if args.log_interval is not None:
        cfg_dict["log_interval"] = args.log_interval

    cfg = SimpleNamespace(**cfg_dict)

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    print(f"Loaded config from {args.config}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    wandb_run = maybe_init_wandb(cfg_dict)

    dataloader = get_dataloader(
        dataset_name=cfg.dataset,
        split="train",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        is_pretrain=True,
        data_root=cfg.data_root,
        img_size=cfg.img_size,
    )

    model = build_model(cfg).to(device)
    criterion = MAELoss(norm_pix_loss=cfg.norm_pix_loss)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    max_steps = int(cfg.max_steps_per_epoch) if cfg.max_steps_per_epoch is not None else len(dataloader)
    grad_accum_steps = int(cfg.grad_accum_steps)
    effective_steps_per_epoch = max(int(math.ceil(max_steps / max(grad_accum_steps, 1))), 1)

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=int(cfg.epochs),
        steps_per_epoch=effective_steps_per_epoch,
        warmup_epochs=int(cfg.warmup_epochs),
        min_lr=float(cfg.min_lr),
        base_lr=float(cfg.lr),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    ckpt_name = f"mae_{cfg.dataset}_vit_tiny_mask{int(cfg.mask_ratio*100)}_dec{cfg.decoder_depth}.pth"
    ckpt_path = os.path.join(cfg.output_dir, ckpt_name)

    for epoch in range(1, cfg.epochs + 1):
        global_step = (epoch - 1) * len(dataloader)
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            patch_size=cfg.patch_size,
            log_interval=cfg.log_interval,
            max_steps_per_epoch=cfg.max_steps_per_epoch,
            wandb_run=wandb_run,
            global_step=global_step,
            amp=bool(cfg.amp),
            grad_accum_steps=int(cfg.grad_accum_steps),
            scaler=scaler,
        )
        print(f"Epoch [{epoch}/{cfg.epochs}]  avg_loss={avg_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss_epoch": avg_loss,
                    "train/epoch": epoch,
                }
            )
        save_checkpoint(model, optimizer, epoch, ckpt_path)

    print(f"Training complete. Checkpoint saved to {ckpt_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
