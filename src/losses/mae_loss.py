import torch.nn as nn

class MAELoss(nn.Module):
    def __init__(self, norm_pix_loss=False):
        super().__init__()
        # storing if we will normalize the patches before loss
        self.norm_pix_loss = norm_pix_loss
    
    def forward(self, pred, target, mask):
        # normalize before loss calculation if needed
        if self.norm_pix_loss: 
            # calucalte mean and variance
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # calculate normalized target
            target = (target-mean) / (var + 1.e-6)**0.5

        # calculate the MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        # avg losss over batch
        loss = (loss * mask).sum() / mask.sum()
        return loss