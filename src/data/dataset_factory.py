# from typing import Optional, Tuple
# import os
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.datasets import CIFAR100, STL10
# from torch.utils.data import Dataset
# from .transform import get_transform

# class ImageOnlyWrapper(Dataset):
#     def __init__(self, base_dataset):
#         self.base = base_dataset

#     def __len__(self):
#         return len(self.base)
    
#     def __getitem__(self, idx):
#         img, _ = self.base[idx]
#         return img
    
# def _tiny_imagenet_dataset(root: str, split: str, transform):
    
#     base = os.path.join(root, 'tiny-imagenet-200')
#     if split == "train":
#         path  = os.path.join(base, "train")
#     elif split in ("val", "test"):
#         path = os.path.join(base, "val")
#     else:
#         raise ValueError(f"Split doesn't work {split}")
#     return datasets.ImageFolder(path, transform=transform)

# def get_dataloader(
#     dataset_name: str,
#     split: str,
#     batch_size: int,
#     num_workers: int = 4,
#     is_pretrain: bool=True,
#     data_root: str = "data",
#     image_size: int = 224,
#     norm_mean: Optional[Tuple[float, float, float]] = None,
#     norm_std: Optional[Tuple[float, float, float]] = None,
#     shuffle: Optional[bool] = None,
#     use_rand_resize_crop: bool = False,
#     use_unlabeled_stl10: bool = True
# ):
#     is_train = split.lower() in ("train", "training")

#     transform = get_transform(
#         image_size = image_size,
#         is_train=is_train,
#         norm_mean=norm_mean,
#         norm_std=norm_std,
#         use_rand_resize_crop=use_rand_resize_crop,
#     )

#     name = dataset_name.lower()
#     data = None

#     if name == "cifar100":
#         data = CIFAR100(root=data_root, train=is_train, download=True, transform=transform)

#     elif name == "stl10":
#         if is_pretrain and use_unlabeled_stl10:
#             stl_split = "unlabeled"
#         else:
#             if is_train:
#                 stl_split = "train"
#             elif split.lower() == "test":
#                 stl_split = "test"
#             else:
#                 stl_split = "test"
            
#         data = STL10(root=data_root, split=stl_split, download=True, transform=transform)
    
#     elif name in ("tiny-imagenet", "tiny_imagenet", "tinyimagenet"):
#         data = _tiny_imagenet_dataset(data_root, split, transform)
    
#     else:
#         raise ValueError(f"{dataset_name} is not supported")
    
#     # Dropping labels during pretraining for MAE
#     if is_pretrain:
#         data = ImageOnlyWrapper(data)

#     # shuffling 
#     if shuffle is None:
#         shuffle = is_train
    
#     return DataLoader(
#         data,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=is_train
#     )