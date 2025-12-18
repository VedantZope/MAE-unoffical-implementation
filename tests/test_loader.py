
import os, sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # one level up from Unit_tests
sys.path.insert(0, str(ROOT))

from src.data.factory import get_dataloader


def check_pretrain():
    loader = get_dataloader(
        dataset_name="cifar100",
        split="train",
        batch_size=8,
        num_workers=0,       # keep 0 while debugging
        is_pretrain=True,
        data_root="data",
        img_size=224,
    )

    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor), f"Pretrain should return Tensor, got {type(batch)}"
    assert batch.ndim == 4, f"Expected 4D tensor (B,C,H,W), got {batch.ndim}D"
    assert batch.shape[0] == 8, f"Expected batch size 8, got {batch.shape[0]}"
    assert batch.shape[1:] == (3, 224, 224), f"Expected (3,224,224), got {tuple(batch.shape[1:])}"
    print("Pretrain loader OK:", tuple(batch.shape))


def check_eval():
    loader = get_dataloader(
        dataset_name="cifar100",
        split="test",
        batch_size=8,
        num_workers=0,
        is_pretrain=False,
        data_root="data",
        img_size=224,
    )

    batch = next(iter(loader))
    assert isinstance(batch, (tuple, list)) and len(batch) == 2, \
        f"Eval should return (images, labels), got {type(batch)} with len {getattr(batch, '__len__', lambda: 'NA')()}"
    imgs, labels = batch

    assert isinstance(imgs, torch.Tensor), f"Images should be Tensor, got {type(imgs)}"
    assert isinstance(labels, torch.Tensor), f"Labels should be Tensor, got {type(labels)}"
    assert imgs.shape == (8, 3, 224, 224), f"Expected imgs (8,3,224,224), got {tuple(imgs.shape)}"
    assert labels.shape == (8,), f"Expected labels (8,), got {tuple(labels.shape)}"
    assert labels.min().item() >= 0 and labels.max().item() < 100, \
        f"Labels out of range for CIFAR-100: min={labels.min().item()} max={labels.max().item()}"

    print("Eval loader OK:", tuple(imgs.shape), tuple(labels.shape),
          f"label_range=[{labels.min().item()}, {labels.max().item()}]")


def check_data_dir():
    if os.path.isdir("data"):
        print("📁 data/ exists")
    else:
        print("data/ does not exist yet (will be created on download)")

def check_tiny_imagenet_if_present():
    base = os.path.join("data", "tiny-imagenet-200")
    if not os.path.isdir(base):
        print("Tiny-ImageNet not found under data/tiny-imagenet-200 (skipping)")
        return

    loader = get_dataloader(
        dataset_name="tiny-imagenet",
        split="val",
        batch_size=4,
        num_workers=0,
        is_pretrain=False,
        data_root="data",
        img_size=224,
    )

    batch = next(iter(loader))
    assert isinstance(batch, (tuple, list)) and len(batch) == 2
    imgs, labels = batch
    assert imgs.shape == (4, 3, 224, 224), f"Expected imgs (4,3,224,224), got {tuple(imgs.shape)}"
    assert labels.shape == (4,), f"Expected labels (4,), got {tuple(labels.shape)}"
    print("Tiny-ImageNet loader OK:", tuple(imgs.shape), tuple(labels.shape))


if __name__ == "__main__":
    print("Running dataloader tests...")
    check_data_dir()
    check_pretrain()
    check_eval()
    check_tiny_imagenet_if_present()
    print("\nALL TESTS PASSED")
