import torch.nn as nn
from .vit_encoder import ViTEncoder
from .mae_decoder import MAEDecoder
from .masking import random_masking

class MAEModel(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        tokens = self.encoder.patch_embed(x)
        # masking
        x_visible, mask, idx_original, idx_keep = random_masking(tokens, self.mask_ratio)
        # encoding
        x_encoded = self.encoder.forward_visible(x_visible, idx_keep)
        x_encoded = x_encoded[:, 1:, :]
        # decoding
        x_reconstructed = self.decoder(x_encoded, idx_keep, idx_original)
        return x_reconstructed, mask
    
    def forward_encoder(self, imgs):
        x_encoded = self.encoder(imgs)
        return x_encoded
