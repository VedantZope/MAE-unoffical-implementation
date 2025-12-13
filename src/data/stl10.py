
from torchvision import datasets


def build_stl10(data_root: str, split: str, transform=None, download: bool = True, is_pretrain: bool = True,):
    
    if split not in ("train", "test"):
        raise ValueError("STL-10 split must be 'train' or 'test'")
    if is_pretrain and split == "train":
        stl_split = "train+unlabeled"
    else:
        stl_split = "train" if split == "train" else "test"
    
    return datasets.STL10(root=data_root,split=stl_split,transform=transform,download=download,)
