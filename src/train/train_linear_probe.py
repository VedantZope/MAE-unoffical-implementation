import argparse
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.factory import get_dataloader
from src.models.vit_encoder import ViTEncoder


def maybe_init_wandb(cfg_dict: Dict[str, Any]):
    if not cfg_dict.get("use_wandb", False):
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[WARN] use_wandb=True but wandb import failed: {e}")
        print("[WARN] Proceeding without wandb logging.")
        return None

    env_mode = os.environ.get("WANDB_MODE")
    mode = env_mode if env_mode is not None else cfg_dict.get("wandb_mode", None)

    try:
        return wandb.init(
            project=cfg_dict.get("wandb_project", "mae-compact"),
            entity=cfg_dict.get("wandb_entity", None),
            name=cfg_dict.get("wandb_run_name", None),
            tags=cfg_dict.get("wandb_tags", None),
            mode=mode,
            config=cfg_dict,
        )
    except Exception as e:
        print(f"[WARN] wandb.init failed ({type(e).__name__}): {e}")
        print("[WARN] Proceeding without wandb logging.")
        return None


def num_classes_for(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name in ("cifar100", "cifar-100", "cifar"):
        return 100
    if name in ("stl10", "stl-10", "stl"):
        return 10
    if name in ("tinyimagenet", "tiny-imagenet", "tiny_imagenet", "tiny"):
        return 200
    raise ValueError(f"Unknown dataset for num_classes: {dataset_name}")


def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model", "mae", "mae_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        # fallback: might already be a raw state dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt  # type: ignore[return-value]
    raise ValueError("Unrecognized checkpoint format; expected a dict containing a state_dict.")


def load_encoder_from_mae_checkpoint(encoder: nn.Module, ckpt_path: str) -> Tuple[int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    enc_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("encoder."):
            enc_state[key[len("encoder.") :]] = v

    if len(enc_state) == 0:
        raise ValueError(
            "No encoder weights found in checkpoint. Expected keys like `encoder.*` in the MAE state_dict."
        )

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    return len(missing), len(unexpected)


class FrozenEncoderLinearProbe(nn.Module):
    def __init__(self, encoder: ViTEncoder, num_classes: int, pool: str = "cls"):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.head = nn.Linear(encoder.norm.normalized_shape[0], num_classes)

        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(imgs)  # (B, 1+N, D)
        if self.pool == "cls":
            return tokens[:, 0, :]
        if self.pool == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        raise ValueError("pool must be 'cls' or 'mean'")

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(imgs)
        return self.head(feats)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        bsz = imgs.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy_top1(logits, labels) * bsz
        total_n += bsz

    return total_loss / max(total_n, 1), total_acc / max(total_n, 1)


def train(cfg: SimpleNamespace):
    device = torch.device(cfg.device)
    print("Device:", device)

    train_loader = get_dataloader(
        dataset_name=cfg.dataset,
        split="train",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        is_pretrain=False,
        data_root=cfg.data_root,
        img_size=cfg.img_size,
    )
    test_loader = get_dataloader(
        dataset_name=cfg.dataset,
        split=cfg.eval_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        is_pretrain=False,
        data_root=cfg.data_root,
        img_size=cfg.img_size,
    )

    encoder = ViTEncoder(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_chans=3,
        embed_dim=cfg.embed_dim,
        depth=cfg.encoder_depth,
        num_heads=cfg.encoder_heads,
        mlp_ratio=cfg.encoder_mlp_ratio,
        dropout=cfg.dropout,
    )
    missing, unexpected = load_encoder_from_mae_checkpoint(encoder, cfg.ckpt_path)
    print(f"Loaded encoder weights from {cfg.ckpt_path} (missing={missing}, unexpected={unexpected})")

    encoder = encoder.to(device)
    encoder.eval()

    num_classes = cfg.num_classes if cfg.num_classes is not None else num_classes_for(cfg.dataset)
    model = FrozenEncoderLinearProbe(encoder=encoder, num_classes=num_classes, pool=cfg.pool).to(device)

    optimizer = torch.optim.SGD(
        model.head.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    wandb_run = maybe_init_wandb(vars(cfg))

    os.makedirs(cfg.output_dir, exist_ok=True)
    run_name = cfg.run_name or f"lprobe_{cfg.dataset}_{cfg.pool}"
    save_path = os.path.join(cfg.output_dir, f"{run_name}.pth")

    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        encoder.eval()

        running_loss = 0.0
        running_acc = 0.0
        total_n = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bsz = imgs.size(0)
            running_loss += loss.item() * bsz
            running_acc += accuracy_top1(logits, labels) * bsz
            total_n += bsz

        scheduler.step()

        train_loss = running_loss / max(total_n, 1)
        train_acc = running_acc / max(total_n, 1)
        test_loss, test_acc = evaluate(model, test_loader, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"{cfg.eval_split} loss {test_loss:.4f} acc {test_acc:.4f} | "
            f"lr {lr:.6f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train/loss": train_loss,
                    "train/acc1": train_acc,
                    f"{cfg.eval_split}/loss": test_loss,
                    f"{cfg.eval_split}/acc1": test_acc,
                }
            )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "head_state": model.head.state_dict(),
                    "best_acc": best_acc,
                    "dataset": cfg.dataset,
                    "pool": cfg.pool,
                    "num_classes": num_classes,
                    "ckpt_path": cfg.ckpt_path,
                    "img_size": cfg.img_size,
                    "patch_size": cfg.patch_size,
                    "embed_dim": cfg.embed_dim,
                },
                save_path,
            )

    print(f"Best {cfg.eval_split} acc: {best_acc:.4f}")
    print(f"Saved best linear head to: {save_path}")
    if wandb_run is not None:
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a linear probe on a frozen MAE encoder (YAML-configured)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p.add_argument("--device", type=str, default=None, help="Optional override for device in YAML")
    p.add_argument("--output-dir", type=str, default=None, help="Optional override for output_dir in YAML")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("PyYAML is required. Install it with `pip install pyyaml`.") from e

    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    default_cfg: Dict[str, Any] = {
        "dataset": "cifar100",
        "data_root": "data",
        "img_size": 224,
        "batch_size": 256,
        "num_workers": 8,
        "eval_split": "test",
        "pool": "cls",
        "epochs": 50,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "num_classes": None,
        "ckpt_path": "",
        "run_name": None,
        "output_dir": "experiments/linear_probe",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dropout": 0.0,
        # model params (must match pretraining encoder)
        "patch_size": 16,
        "embed_dim": 192,
        "encoder_depth": 12,
        "encoder_heads": 3,
        "encoder_mlp_ratio": 4.0,
        # tracking
        "use_wandb": False,
        "wandb_project": "mae-compact",
        "wandb_entity": None,
        "wandb_run_name": None,
        "wandb_tags": None,
        "wandb_mode": None,
    }

    cfg_dict = {**default_cfg, **yaml_cfg}
    if args.device is not None:
        cfg_dict["device"] = args.device
    if args.output_dir is not None:
        cfg_dict["output_dir"] = args.output_dir

    if not cfg_dict.get("ckpt_path"):
        raise ValueError("Config must set `ckpt_path` to a pretrained MAE checkpoint.")

    cfg = SimpleNamespace(**cfg_dict)
    train(cfg)


if __name__ == "__main__":
    main()
