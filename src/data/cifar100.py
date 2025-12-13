from torchvision import datasets

# call cifar100 and put the automatic split to train
def build_cifar100(data_root: str, split: str, transform=None, download: bool = True):
    if split not in ("train", "test"):
        raise ValueError("CIFAR-100 split must be 'train' or 'test'")
    train = (split == "train")
    return datasets.CIFAR100(
        root=data_root,
        train=train,
        transform=transform,
        download=download,
    )