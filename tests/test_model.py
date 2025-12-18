import sys
import os
import torch
import pytest
import os, sys
import torch
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))

from src.models.mae import MAEModel
from src.models.mae_decoder import MAEDecoder
from src.models.vit_encoder import ViTEncoder

def test_mae_forward():
    images = torch.randn(2, 3, 224, 224)
    encoder = ViTEncoder(img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12)
    decoder = MAEDecoder(embed_dim=768, decoder_dim=512, num_patches=(224 // 16) ** 2, depth=4, num_heads=8)
    model = MAEModel(encoder, decoder, mask_ratio=0.75)
    out, mask = model(images)
    assert out.shape[0] == images.shape[0]
    assert mask.shape[0] == images.shape[0]
    print("MAE forward test passed.")

if __name__ == "__main__":
    test_mae_forward()