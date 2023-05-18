import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(Module):
    """ Custom masked MSE loss"""

    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predicted, target, mask):
        diff = predicted - target
        masked_diff = diff * mask
        loss_value = torch.mean(masked_diff ** 2)
        return loss_value

class NavierStokesLoss(nn.Module):
    def __init__(self):
        super(NavierStokesLoss, self).__init__()

        # Finite difference kernels for dx and dy (centered differences)
        self.dx_kernel = torch.Tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]]).float()
        self.dy_kernel = torch.Tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]]).float()

    def forward(self, preds):
        u, v = preds[:, 0], preds[:, 1]  # assuming u and v are the first two channels

        # Calculate gradients
        du_dx = F.conv2d(u.unsqueeze(1), self.dx_kernel, padding=1)
        du_dy = F.conv2d(u.unsqueeze(1), self.dy_kernel, padding=1)
        dv_dx = F.conv2d(v.unsqueeze(1), self.dx_kernel, padding=1)
        dv_dy = F.conv2d(v.unsqueeze(1), self.dy_kernel, padding=1)

        # Momentum equations
        momentum_u = torch.abs(u * du_dx + v * du_dy)
        momentum_v = torch.abs(u * dv_dx + v * dv_dy)

        # Continuity equation
        continuity = torch.abs(du_dx + dv_dy)

        return (continuity + momentum_u + momentum_v).mean()


