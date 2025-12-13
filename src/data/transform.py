# from typing import Tuple, Optional
# from torchvision import transforms

# def get_transform(
#     image_size: int = 224,
#     is_train: bool = True,
#     norm_mean: Optional[Tuple[float, float, float]] = None,
#     norm_std: Optional[Tuple[float, float, float]] = None,
#     rand_crop: bool = True,
#     use_rand_resize_crop: bool = False,
#     flip: bool = True
# ):
    
#     # calculate mean and standard deviation ofr normalization 
#     mean = norm_mean
#     std = norm_std 

#     # implementing training mode where we transform the images
#     if is_train:
#         ops = []
    
#         if flip:
#             # flipping image with equal probability
#             ops += [transforms.RandomHorizontalFlip()]
#         # MAE method of cropping using random size and aspect ratio for mask
#         if use_rand_resize_crop:
#             ops += [transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0))]
#         else:
#             # apply the standard approach of resizing and then cropping
#             # random cropping to the desired size

#             if rand_crop:
#                 # upsample and then do random cropping
#                 ops += [transforms.Resize(image_size + 32), transforms.RandomCrop(image_size)]
#             else:
#                 # deterministic cropping by resizing and cropping around center
#                 ops += [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
        
#     else:
#         # Evaluation mode is just center cropping and resizing
#         ops = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    
#     # Returning as a tensor (C, H, W in [0,1]) normalizing with mean and stdev
#     ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
#     return transforms.Compose(ops)

