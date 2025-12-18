
from typing import Literal
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .cifar100 import build_cifar100
from .stl10 import build_stl10
from .tiny_imagenet import build_tiny_imagenet


# mean std for imagenet -> Found online
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ImagesOnly(Dataset): # for pretrained
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)
        
    def __getitem__(self, idx):
        x, _ = self.base_ds[idx]
        return x


def _build_transforms(is_pretrain: bool, img_size: int = 224):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if is_pretrain:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def _build_dataset(
    dataset_name: str,
    split: Literal["train", "test", "val"],
    data_root: str,
    transform,
    is_pretrain: bool,
):
    name = dataset_name.lower()

    # map 'val' to 'test' for datasets that only have train/test
    if split == "val":
        split = "test"

    if name in ("cifar100", "cifar-100", "cifar"):
        return build_cifar100(
            data_root=data_root, 
            split=split, 
            transform=transform, 
            download=True
        )
    if name in ("stl10", "stl-10", "stl"):
        return build_stl10(
            data_root=data_root,
            split=split,
            transform=transform,
            download=True,
            is_pretrain=is_pretrain,
        )
    if name in ("tinyimagenet", "tiny-imagenet", "tiny_imagenet", "tiny"):
        return build_tiny_imagenet(
            data_root=data_root,
            split=split,
            transform=transform,
        )

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def get_dataloader(
    dataset_name: str,
    split: Literal["train", "test", "val"],
    batch_size: int,
    num_workers: int,
    is_pretrain: bool = True,
    data_root: str = "data",
    img_size: int = 224,
    pin_memory: bool = True,
):
    # is_pretrain =  True (returns images) & is_pretrain=False (returns image, label)
    tfm = _build_transforms(is_pretrain=is_pretrain, img_size=img_size)
    ds = _build_dataset(dataset_name, split, data_root, tfm, is_pretrain=is_pretrain)

    if is_pretrain:
        ds = ImagesOnly(ds)

    shuffle = (split == "train")
    drop_last = (split == "train")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
