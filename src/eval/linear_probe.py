import torch
import torch.nn as nn


class LinearProbeHead(nn.Module):
    def __init__(self, mae_model, embed_dim: int, num_classes: int, pool: str = "cls"):
        super().__init__()
        self.mae = mae_model
        self.pool = pool
        self.head = nn.Linear(embed_dim, num_classes)

        # Freeze encoder
        for p in self.mae.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_features(self, imgs: torch.Tensor) -> torch.Tensor:
        # MAEModel.forward_encoder returns encoder(imgs) :contentReference[oaicite:3]{index=3}
        tokens = self.mae.forward_encoder(imgs)

        if self.pool == "cls":
            # ViTEncoder prepends cls token :contentReference[oaicite:4]{index=4}
            feats = tokens[:, 0, :]
        elif self.pool == "mean":
            feats = tokens[:, 1:, :].mean(dim=1)
        else:
            raise ValueError("pool must be 'cls' or 'mean'")

        return feats

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(imgs)
        logits = self.head(feats)
        return logits
