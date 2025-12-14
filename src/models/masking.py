import torch

def random_masking(x, mask_ratio):
    B, N, D = x.shape
    # number of tokens to keep
    tokens_keep = int(N * (1 - mask_ratio))
    # Creating noise for each sample
    noise = torch.rand(B, N, device = x.device)

    # shuffle index for noise
    idx_shuffle = torch.argsort(noise, dim=1)

    # Returning indices to original order post masking
    idx_original = torch.argsort(idx_shuffle, dim=1)

    # Select indices to keep
    idx_keep = idx_shuffle[:, :tokens_keep]

    # Getting visibile tokens
    x_visible = torch.gather(x, dim=1, index=idx_keep.unsqueeze(-1).expand(-1, -1, D))

    # creating the mask where 1 is masked and 0 is unaltered
    mask = torch.ones([B, N], device=x.device)
    mask[:, :tokens_keep] = 0
    mask = torch.gather(mask, dim=1, index=idx_original)

    return x_visible, mask, idx_original, idx_keep