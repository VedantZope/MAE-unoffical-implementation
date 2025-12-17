import os
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.vit_encoder import ViTEncoder
from src.models.mae_decoder import MAEDecoder
from src.models.mae import MAEModel
from src.eval.linear_probe import LinearProbeHead
from src.data.factory import get_dataloader


def num_classes_for(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name in ("cifar100", "cifar-100"):
        return 100
    if name in ("stl10", "stl-10"):
        return 10
    raise ValueError(f"Unknown dataset for num_classes: {dataset_name}")


def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def load_mae_checkpoint(mae: MAEModel, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Common patterns:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "mae", "mae_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    missing, unexpected = mae.load_state_dict(ckpt, strict=False)
    if len(missing) > 0:
        print(f"[WARN] Missing keys when loading checkpoint (showing up to 10): {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys when loading checkpoint (showing up to 10): {unexpected[:10]}")


def build_mae_from_args(args) -> MAEModel:
    encoder = ViTEncoder(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.enc_depth,
        num_heads=args.enc_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )

    # Decoder is not used for probing, but MAEModel expects it.
    # Keep it consistent with your constructor signatures.
    num_patches = (args.img_size // args.patch_size) ** 2
    decoder = MAEDecoder(
        embed_dim=args.embed_dim,
        decoder_dim=args.dec_dim,
        num_patches=num_patches,
        depth=args.dec_depth,
        num_heads=args.dec_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )

    mae = MAEModel(encoder=encoder, decoder=decoder, mask_ratio=args.mask_ratio)
    return mae


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    criterion = nn.CrossEntropyLoss()

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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    # Dataloaders must be in supervised mode (is_pretrain=False) so they yield (img, label)
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_pretrain=False,
        data_root=args.data_root,
        img_size=args.img_size,
    )
    test_loader = get_dataloader(
        dataset_name=args.dataset,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_pretrain=False,
        data_root=args.data_root,
        img_size=args.img_size,
    )

    mae = build_mae_from_args(args)

    if args.ckpt:
        print("Loading checkpoint:", args.ckpt)
        load_mae_checkpoint(mae, args.ckpt)

    mae.to(device)
    mae.encoder.eval()  # frozen encoder; also disables dropout for stable features

    n_classes = args.num_classes if args.num_classes is not None else num_classes_for(args.dataset)
    probe = LinearProbeHead(mae_model=mae, embed_dim=args.embed_dim, num_classes=n_classes, pool=args.pool).to(device)

    # Only optimize the linear head
    optimizer = torch.optim.SGD(
        probe.head.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"linear_probe_{args.dataset}_{args.pool}.pth")

    for epoch in range(1, args.epochs + 1):
        probe.train()   # head trains
        mae.encoder.eval()  # encoder stays frozen + deterministic

        running_loss = 0.0
        running_acc = 0.0
        total_n = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bsz = imgs.size(0)
            running_loss += loss.item() * bsz
            running_acc += accuracy_top1(logits, labels) * bsz
            total_n += bsz

        train_loss = running_loss / max(total_n, 1)
        train_acc = running_acc / max(total_n, 1)

        test_loss, test_acc = evaluate(probe, test_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "probe_head": probe.head.state_dict(),
                    "dataset": args.dataset,
                    "pool": args.pool,
                    "embed_dim": args.embed_dim,
                    "num_classes": n_classes,
                    "best_acc": best_acc,
                },
                save_path,
            )

    print(f"Best test acc: {best_acc:.4f}")
    print(f"Saved best linear head to: {save_path}")


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)

    # checkpoint
    p.add_argument("--ckpt", type=str, default="", help="Path to MAE checkpoint (.pth).")

    # probe
    p.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_classes", type=int, default=None)

    # model (must match what was pretrained)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--enc_depth", type=int, default=12)
    p.add_argument("--enc_heads", type=int, default=12)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)

    # decoder args (not used for probing, but needed to build MAEModel consistently)
    p.add_argument("--dec_dim", type=int, default=512)
    p.add_argument("--dec_depth", type=int, default=4)
    p.add_argument("--dec_heads", type=int, default=8)
    p.add_argument("--mask_ratio", type=float, default=0.75)

    # misc
    p.add_argument("--out_dir", type=str, default="experiments/linear_probe")
    p.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    if args.ckpt == "":
        args.ckpt = None
    return args


if __name__ == "__main__":
    train(parse_args())
