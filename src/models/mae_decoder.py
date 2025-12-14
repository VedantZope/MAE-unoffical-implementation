import torch
import torch.nn as nn

class MAEDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim=768, 
            decoder_dim=512, 
            num_patches=196, 
            depth=4, 
            num_heads=8, 
            mlp_ratio=4.0, 
            dropout=0.1
    ):
        super().__init__()
        # mask token for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        # positional embedding for batches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        # project encoder output to decoder 
        self.proj = nn.Linear(embed_dim, decoder_dim) if embed_dim != decoder_dim else nn.Identity()
        # transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio), 
            dropout=dropout, 
            activation="gelu", 
            batch_first= True
            )
        # setting the decoder
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        # linear head to map output to pixels
        self.head = nn.Linear(decoder_dim, 3 * 16 * 16)

    def forward(self, x_visible, idx_keep, idx_original):
        # proj decoder dims
        x_visible = self.proj(x_visible)
        B, N_visible, C_dec = x_visible.shape

        # for n patches
        N = self.pos_embed.shape[1]
        x_full = self.mask_token.repeat(B, N, 1)

        # scatter tokens that are visible to original pos
        # use expand to match decoder dim for indexing
        x_full.scatter_(1, idx_keep.unsqueeze(-1).expand(-1, -1, C_dec), x_visible)

        # add pos embeddings
        x_full = x_full + self.pos_embed

        # decode: map to pix
        x_full = self.decoder(x_full)
        x_reconstructed = self.head(x_full)
        return x_reconstructed
