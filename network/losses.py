import torch
from torch.nn import Module


class MaskedMSELoss(Module):
    """ Custom masked MSE loss"""

    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predicted, target, mask):
        diff = predicted - target
        masked_diff = diff * mask
        loss_value = torch.mean(masked_diff ** 2)
        return loss_value


