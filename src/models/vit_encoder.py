import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # gettting the number of patches
        self.grid_size = (img_size // patch_size) ** 2
        # extract and embed patches with conv2d
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # input: (B, C, H, W) 
        # Output: (B, embeded_dims, H/patch size, W/ patch size)
        x = self.proj(x)
        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    # TODO: We are discussing this section on how to implement this
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        # patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # learnable token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.grid_size, embed_dim)
        )
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # layer normalization on output
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, _ = x.shape
        # class tokens for each batch
        cls_tokens = self.cls_token.expand(B, 1, -1)
        # concat class tokens
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedddings encode and normalize
        x = x + self.pos_embed[:, :(N+1), :]
        x = self.encoder(x)
        x = self.norm(x)
        return x
    
    def forward_tokens(self, tokens):
        B, N, _ = tokens.shape
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + self.pos_embed[:, :(N+1), :]
        x = self.encoder(x)
        x = self.norm(x)
        return x
