import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:,:,1:,:])
        count_w = self.tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class MaskedMSELoss(torch.nn.Module):
    """ Custom pixel-wise MSE loss"""
    def __init__(self, device):
        super(MaskedMSELoss, self).__init__()
        self.device = device

    def forward(self, predicted, target, mask):
        predicted = predicted.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)  # Mask is not used for the loss calculation

        # Calculate the difference between predicted and target
        diff = predicted - target

        # Pixel-wise mean of the squared difference
        loss_value = torch.mean(diff ** 2)

        return loss_value


class NavierStokesLoss(nn.Module):
    def __init__(self, device):
        super(NavierStokesLoss, self).__init__()

        # Finite difference kernels for dx and dy (centered differences)
        self.dx_kernel = torch.Tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]]).float().to(device)
        self.dy_kernel = torch.Tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]]).float().to(device)

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

class GraphNavierStokesLoss(torch.nn.Module):
    def __init__(self):
        super(GraphNavierStokesLoss, self).__init__()

    def forward(self, data):
        u, v, x, y = data.x[:, 0], data.x[:, 1], data.x[:, 4], data.x[:, 5]

        EPSILON = 1e-7  # Small constant to avoid division by zero

        # Compute differences in velocities and positions for each edge
        du = u[data.edge_index[0]] - u[data.edge_index[1]]
        dv = v[data.edge_index[0]] - v[data.edge_index[1]]
        dx = x[data.edge_index[0]] - x[data.edge_index[1]]
        dy = y[data.edge_index[0]] - y[data.edge_index[1]]

        # Calculate du/dx and dv/dy
        du_dx = du / (dx.abs() + EPSILON)  # Take absolute value of dx
        dv_dy = dv / (dy.abs() + EPSILON)  # Take absolute value of dy

        # Compute the divergence for each edge
        div_edge = du_dx + dv_dy

        # Now we want to sum all the divergence contributions for each node
        div_node = torch_scatter.scatter_add(div_edge.abs(), data.edge_index[0], dim=0, dim_size=data.num_nodes)

        # Return the mean divergence
        return div_node.mean()








